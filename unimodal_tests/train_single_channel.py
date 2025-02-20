import os
import sys
from typing import Optional, List
import time
from copy import deepcopy
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "diverse_channel_vit"))

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR

from diverse_channel_vit.config import MyConfig
from diverse_channel_vit.trainer import Trainer
from unimodal_tests.datasets.single_channel_dataset import SingleChannelDataset
from diverse_channel_vit.datasets.dataset_utils import get_mean_std_dataset
from diverse_channel_vit import utils  # Use the main utils module
from diverse_channel_vit.morphem.benchmark import run_benchmark
from diverse_channel_vit.datasets.tps_transform import TPSTransform, dummy_transform  # Add TPS transform imports

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)

def ddp_setup():
    print(f"-------------- Setting up ddp {os.environ['LOCAL_RANK']}...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_transforms(modality, channel_idx, is_train=True, img_size=224):
    """Get transforms matching original code's normalization."""
    # Get modality-specific mean and std
    mean_std = get_mean_std_dataset(modality)
    if isinstance(mean_std, tuple):
        mean, std = mean_std
    else:
        mean, std = mean_std[modality]
    
    # Use only the specific channel's normalization
    channel_mean = [mean[channel_idx]]
    channel_std = [std[channel_idx]]
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(channel_mean, channel_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.Normalize(channel_mean, channel_std),
        ])
    return transform

def get_channel_name_from_idx(modality, channel_idx):
    """Get channel name from index based on SingleChannelDataset.CHANNEL_MAPS."""
    channel_map = SingleChannelDataset.CHANNEL_MAPS[modality]
    for name, idx in channel_map.items():
        if idx == channel_idx:
            return name
    raise ValueError(f"No channel found for index {channel_idx} in modality {modality}")

def get_single_channel_loaders(cfg, modality, channel_idx, is_train=True):
    """Create dataloaders for a single channel."""
    channel_name = get_channel_name_from_idx(modality, channel_idx)
    print(f"DEBUG: Creating loader for {modality} - Channel {channel_idx} ({channel_name})")
    transform = get_transforms(modality, channel_idx, is_train=is_train, img_size=cfg.dataset.img_size)
    
    dataset = SingleChannelDataset(
        csv_path=os.path.join(cfg.dataset.root_dir, cfg.dataset.file_name),
        modality=modality,
        channel_name=channel_name,
        root_dir=cfg.dataset.root_dir,
        is_train=is_train,
        transform=transform
    )
    
    if cfg.hardware.multi_gpus == "ddp" and is_train:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = is_train
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.train.batch_size if is_train else cfg.eval.batch_size,
        shuffle=shuffle,
        num_workers=cfg.hardware.num_workers,
        sampler=sampler,
        pin_memory=True
    )
    
    return loader

class SingleChannelTrainer(Trainer):
    # Channel index mapping from original dataset to trainer's indices
    CHANNEL_INDEX_MAP = {
        "Allen": {0: 0},  # DNA: original 0 -> trainer's 0 (simplified mapping)
        "CP": {4: 0}      # Membrane: original 4 -> trainer's 0 (simplified mapping)
    }
    
    def _build_dataset(self):
        """Override dataset building to use SingleChannelDataset."""
        # Get channel indices from config using original dataset mapping
        source_modality = list(self.cfg.data_chunk.chunks[0].keys())[0]  # 'Allen'
        source_channel_idx = self.cfg.data_chunk.chunks[0][source_modality][0]  # 0 for DNA
        target_modality = list(self.cfg.data_chunk.chunks[1].keys())[0]  # 'CP'
        target_channel_idx = self.cfg.data_chunk.chunks[1][target_modality][0]  # 4 for Membrane
        
        print(f"DEBUG: Setting up datasets for source={source_modality}(ch={source_channel_idx}) and target={target_modality}(ch={target_channel_idx})")
        print(f"DEBUG: Source channel will be {get_channel_name_from_idx(source_modality, source_channel_idx)} from {source_modality}")
        print(f"DEBUG: Target channel will be {get_channel_name_from_idx(target_modality, target_channel_idx)} from {target_modality}")
        
        # Source channel (training)
        self.train_loaders[source_modality] = get_single_channel_loaders(
            self.cfg, source_modality, source_channel_idx, is_train=True
        )
        self.val_loaders[source_modality] = get_single_channel_loaders(
            self.cfg, source_modality, source_channel_idx, is_train=False
        )
        
        # Target channel (testing)
        target_loader = get_single_channel_loaders(
            self.cfg, target_modality, target_channel_idx, is_train=False
        )
        
        # Set up test loaders for both source and target modalities
        self.test_loaders[source_modality] = get_single_channel_loaders(
            self.cfg, source_modality, source_channel_idx, is_train=False
        )
        self.test_loaders[target_modality] = target_loader
        
        # Set up classes (using original class mappings)
        self.data_classes_train = SingleChannelDataset.CLASSES[source_modality]
        self.data_classes_test = SingleChannelDataset.CLASSES[target_modality]
        
        # Set up mapper using trainer's indices
        self.mapper = {
            source_modality: [self.CHANNEL_INDEX_MAP[source_modality][source_channel_idx]],  # Map to 0
            target_modality: [self.CHANNEL_INDEX_MAP[target_modality][target_channel_idx]]   # Map to 0
        }
        print(f"DEBUG: Mapper configuration: {self.mapper}")
        
        # Required for training setup
        self.train_loaders[self.shuffle_all] = self.train_loaders[source_modality]
        self.num_loaders = len(self.cfg.data_chunk.chunks)
        
        # Set up data channels
        self.data_channels = {
            source_modality: [source_channel_idx],
            target_modality: [target_channel_idx]
        }
        print(f"DEBUG: Data channels configuration: {self.data_channels}")
        
        # Print sample batch information
        for name, loader in self.train_loaders.items():
            batch = next(iter(loader))
            if isinstance(batch, (tuple, list)):
                x, y = batch
                print(f"DEBUG: {name} loader - X shape: {x.shape}, Y shape: {y.shape}")
            else:
                x = batch["image"] if "image" in batch else batch[0]
                y = batch["label"] if "label" in batch else batch[1]
                print(f"DEBUG: {name} loader - X shape: {x.shape}, Y shape: {y.shape}")
    
    def _forward_model(self, x, chunk_name, training_chunks=None, init_first_layer=None, new_channel_init=None, **kwargs):
        """Override forward model to add debugging information."""
        return super()._forward_model(x, chunk_name, training_chunks, init_first_layer, new_channel_init, **kwargs)

    @torch.inference_mode()
    def eval_morphem70k(self, epoch: int, new_channel_init: str, eval_chunks: Optional[List[str]] = None):
        """Override eval_morphem70k to handle single-channel evaluation."""
        self.logger.info(f"Start evaluation for epoch {epoch} with new_channel_init={new_channel_init}")
        self.model.eval()

        training_chunks = self.cfg.train.training_chunks
        init_first_layer = self.cfg.model.init_first_layer

        eval_cfg = deepcopy(self.cfg.eval)
        scc_jobid = utils.default(self.cfg.logging.scc_jobid, "")
        FOLDER_NAME = f'{utils.datetime_now("%Y-%b-%d")}_seed{self.cfg.train.seed}_sccid{scc_jobid}'
        eval_cfg.dest_dir = os.path.join(eval_cfg.dest_dir.format(FOLDER_NAME=FOLDER_NAME), f"epoch_{epoch}")
        utils.mkdir(eval_cfg.dest_dir)
        utils.mkdir(eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME))
        eval_cfg.feature_dir = eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME)

        start_time = time.time()
        out_path_list = []
        
        # Use configured eval_chunks or default to our chunks
        if eval_chunks is None:
            eval_chunks = ["Allen", "CP"]  # Only evaluate on Allen and CP
            
        for chunk_name in eval_chunks:
            if chunk_name not in self.test_loaders:
                continue  # Skip if we don't have a loader for this chunk
                
            feat_outputs = []
            eval_loader = self.test_loaders[chunk_name]
            print(f"Start getting features for {chunk_name}...")
            
            for bid, batch in enumerate(eval_loader):
                # Handle both training and evaluation batch formats
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                output = self._forward_model(
                    x,
                    chunk_name=chunk_name,
                    training_chunks=training_chunks,
                    init_first_layer=init_first_layer,
                    new_channel_init=new_channel_init,
                )
                feat_outputs.append(output)

            feat_outputs = torch.cat(feat_outputs, dim=0).cpu().numpy()
            # Save features directly in the features directory under the chunk name
            folder_path = os.path.join(eval_cfg.feature_dir, chunk_name)
            utils.mkdir(folder_path)
            out_path = os.path.join(folder_path, eval_cfg.feature_file)
            out_path_list.append(out_path)
            np.save(out_path, feat_outputs)
            runtime = round((time.time() - start_time) / 60, 2)
            print(f"-- Done writing features for {chunk_name} in total {runtime} minutes")

        # Run benchmark only on the chunks we have
        torch.cuda.empty_cache()
        cosine_metrics = None
        for classifier in eval_cfg.classifiers:
            eval_cfg.classifier = classifier
            if classifier == "knn":
                for knn_metric in eval_cfg.knn_metrics:
                    if "cosine" in knn_metric:
                        # Run benchmark on each chunk separately
                        all_results = []
                        for chunk_name in eval_chunks:
                            print(f"Running benchmark for {chunk_name}...")
                            full_res = run_benchmark(
                                eval_cfg["root_dir"],
                                eval_cfg["dest_dir"],
                                eval_cfg["feature_dir"],
                                eval_cfg["feature_file"],
                                eval_cfg["classifier"],
                                False,  # No UMAP for single-channel
                                eval_cfg["use_gpu"],
                                knn_metric
                            )
                            all_results.append(full_res)
                        
                        if all_results:
                            cosine_metrics = pd.concat(all_results, ignore_index=True)
                            # Process metrics as before...
                            cosine_metrics["key"] = cosine_metrics.iloc[:, 0:3].apply(lambda x: "/".join(x.astype(str)), axis=1)
                            acc = dict(zip(cosine_metrics["key"] + f"/{knn_metric}/acc", cosine_metrics["accuracy"] * 100))
                            f1 = dict(zip(cosine_metrics["key"] + f"/{knn_metric}/f1", cosine_metrics["f1_score_macro"]))
                            
                            metrics_logger = {
                                **acc,
                                **f1,
                                f"{eval_cfg.classifier}/{knn_metric}/score_acc/": np.mean(list(acc.values())[1:]),
                                f"{eval_cfg.classifier}/{knn_metric}/score_f1/": np.mean(list(f1.values())[1:])
                            }
                            self.logger.info(metrics_logger, sep="| ", padding_space=True)

        if self.cfg.eval.clean_up:
            for out_path in out_path_list:
                os.remove(out_path)
            self.logger.info(f"cleaned up {len(out_path_list)} files after evaluation")
            
        return cosine_metrics

@hydra.main(version_base=None, config_path="configs", config_name="single_channel_test")
def main(cfg: MyConfig) -> None:
    use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()

    if use_ddp:
        ddp_setup()

    num_gpus = torch.cuda.device_count()
    if "num_gpus" not in cfg["hardware"]:
        OmegaConf.update(cfg, "hardware.num_gpus", num_gpus, force_add=True)

    trainer = SingleChannelTrainer(cfg)
    print("Starting trainer.train()...")
    trainer.train()

    if use_ddp:
        destroy_process_group()

if __name__ == "__main__":
    main() 