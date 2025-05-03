import os
import sys
import torch
from collections import OrderedDict

project_root = os.path.dirname(os.path.abspath(__file__))
print(project_root)
vit_dir = os.path.join(project_root, "diverse_channel_vit")
sys.path.insert(0, vit_dir)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from diverse_channel_vit.trainer import Trainer
import torch

# === Channel subset setup ===
# subset_indices_within_cp = [0, 4, 5]  # Channels relative to full input
# cp_mapper = [5, 0, 7, 1, 4]           # CP chunk full input channel indices
# selected_channels = [cp_mapper.index(i) for i in subset_indices_within_cp]
subset_indices_within_cp = [7, 11, 10]  # for example
cp_mapper = [7, 8, 9, 10, 11]  # current CP
selected_channels = [cp_mapper.index(i) for i in subset_indices_within_cp]  # → [0, 4, 3]

# === Hydra config loading ===
#config_dir = os.path.join(project_root, "configs")
with initialize_config_dir(config_dir="/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/configs", version_base=None):
    overrides = [
        "model=dichavit",
        #"data_chunk=morphem70k",
        "data_chunk=CP",
        "dataset=adaptive_morphem70k_v2_12channels",
        "logging=no",
        "hardware=default",
        "eval=default",
        "optimizer=adamw",
        "train=random_instance",
        "scheduler=none"
    ]
    cfg = compose(config_name="chammi_cfg", overrides=overrides)
OmegaConf.set_struct(cfg, False)

if not hasattr(cfg.data_chunk, "chunks") and hasattr(cfg.data_chunk, "chunk"):
    cfg.data_chunk.chunks = [{cfg.data_chunk.chunk: cfg.dataset.in_channel_names}]

# === Modify config for subset eval ===
cfg.eval.channel_combinations = selected_channels
cfg.eval.dest_dir = f"/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/results/subset_eval_baseline{'_'.join(map(str, selected_channels))}"
cfg.eval.use_gpu = torch.cuda.is_available()
trainer = Trainer(cfg)

# === Load model and run eval ===
#checkpoint_path = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt"
checkpoint_path = "/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/checkpoints/morphem70k/2025-Mar-14-10-33-06--seed2025/model_last.pt"


# trainer._load_model(checkpoint_path)
# trainer.eval_morphem70k(epoch=60, new_channel_init="")  # change if needed

# === Load checkpoint manually and clean keys ===
checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

# Strip "module." prefix from keys if model was trained using DataParallel or DDP
clean_state_dict = {
    k.replace("module.", ""): v for k, v in checkpoint["model_params"].items()
}

# Load into model with strict=False to tolerate any mismatches
missing, unexpected = trainer.model.load_state_dict(clean_state_dict, strict=False)

print(f"Model weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)
    
trainer.model.feature_extractor.current_selected_channels = selected_channels
trainer.model.feature_extractor.current_chunk_name = "CP"
trainer.model.feature_extractor.current_epoch = 60

#trainer.model.feature_extractor.mapper["CP"] = subset_indices_within_cp
trainer.model.feature_extractor.patch_embed.mapper["CP"] = subset_indices_within_cp
# === Run evaluation ===
trainer.eval_morphem70k(epoch=60, new_channel_init="",  eval_chunks=["CP"])
#trainer.eval_morphem70k(chunk="CP", epoch=60, new_channel_init="")
print("Evaluating chunk:", cfg.data_chunk)  # or whatever field stores that


print("ubset evaluation complete!")
