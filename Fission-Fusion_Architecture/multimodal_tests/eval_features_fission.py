import os
import sys
import torch
from collections import OrderedDict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
vit_dir = os.path.join(project_root, "diverse_channel_vit")
sys.path.insert(0, vit_dir)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from diverse_channel_vit.trainer import Trainer

# === Load Hydra Config ===
config_dir = "/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/diverse_channel_vit/configs"

with initialize_config_dir(config_dir=config_dir, version_base=None):
    overrides = [
        "model=dichavit",  # 
        "dataset=morphem70k_v2_12channels",  # FULL dataset
        "data_chunk=morphem70k",  # FULL dataset
        "logging=no",
        "hardware=default",
        "eval=default",
        "optimizer=adamw",
        "train=random_instance",
        "scheduler=none",
        "model.block_type=block_v2",
        "model.use_fission_module=True",
        "model.use_parallel_paths=True",
        "model.share_transformer_weights=True",
        "model.use_fusion_module=True",
        "model.fission_hidden_dim=512",
        "model.fission_shared_dim=384",
        "model.fission_specific_dim=384",
        "model.feature_separation_lambda=0.1",
        "model.adversarial_lambda=0.05",
         
    ]
    overrides.append("model.dropout_tokens_hcs=none") 
    cfg = compose(config_name="chammi_cfg", overrides=overrides)
OmegaConf.set_struct(cfg, False)

# ===  fix to config ===
if not hasattr(cfg.data_chunk, "chunks") and hasattr(cfg.data_chunk, "chunk"):
    cfg.data_chunk.chunks = [{cfg.data_chunk.chunk: cfg.dataset.in_channel_names}]

cfg.eval.use_gpu = torch.cuda.is_available()

# You can optionally change dest_dir if you want to save into a new folder
cfg.eval.dest_dir = "/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/separate_features"

# === Initialize trainer ===
trainer = Trainer(cfg)

# === Load Checkpoint ===
checkpoint_path = "/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/checkpoints/morphem70k/2025-Apr-27-11-10-22--seed2025/model_last.pt"
#checkpoint_path = "/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/checkpoints/morphem70k/2025-Apr-27-11-09-57--seed2025/model_last.pt"

checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

# Strip "module." if trained with DataParallel
clean_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_params"].items()}
keys_to_delete = [k for k in clean_state_dict.keys() if "proxies" in k or "adaptive_interface" in k]
for k in keys_to_delete:
    print(f"Deleting key from state_dict: {k}")
    del clean_state_dict[k]

missing, unexpected = trainer.model.load_state_dict(clean_state_dict, strict=False)

print(f"Model weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

print(trainer.model.feature_extractor.fission_module.shared_mlp[0].weight.abs().mean())
print(trainer.model.feature_extractor.fission_module.specific_mlp[0].weight.abs().mean())

# === Evaluate ===
# VERY IMPORTANT: pass return_shared_specific=True inside your model already

trainer.eval_morphem70k(epoch=60, new_channel_init="zero")

print("Evaluation complete!")
