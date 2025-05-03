import os
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from diverse_channel_vit.trainer import Trainer
from tqdm import tqdm
import sys 

# Get project root
project_root = os.path.dirname(os.path.abspath(__file__))
print("Project Root:", project_root)

# Add project root to sys.path so modules like 'models' can be found
sys.path.insert(0, project_root)

# Add diverse_channel_vit separately (optional, but may help with relative imports)
vit_dir = os.path.join(project_root, "diverse_channel_vit")
sys.path.insert(0, vit_dir)

# === Define script logic ===

# Load config
config_dir = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/configs"
checkpoint_path = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt"
chunk_name = "CP"
subset_global_indices = [7, 11, 10]  # subset from CHAMMI full input
cp_mapper = [7, 8, 9, 10, 11]
selected_channels = [cp_mapper.index(i) for i in subset_global_indices]  # [0, 4, 3]

# Initialize Hydra config
with initialize_config_dir(config_dir=config_dir, version_base=None):
    overrides = [
        "model=dichavit_adaptive",
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

cfg.eval.use_gpu = torch.cuda.is_available()
trainer = Trainer(cfg)

# === Load model ===
checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
clean_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_params"].items()}
missing, unexpected = trainer.model.load_state_dict(clean_state_dict, strict=False)

# === Apply subset channel info ===
trainer.model.feature_extractor.patch_embed.current_selected_channels = subset_global_indices
trainer.model.feature_extractor.patch_embed.current_chunk_name = chunk_name
trainer.model.feature_extractor.patch_embed.current_epoch = 60

# === Setup dataloader ===
dataloader = trainer.test_loader[chunk_name]

# === Run eval ===
trainer.model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating"):
        x, y = batch["image"], batch["label"]
        if cfg.eval.use_gpu:
            x, y = x.cuda(), y.cuda()

        outputs = trainer.model(x, chunk_name=chunk_name)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

# Save predictions or compute accuracy externally
import torch.nn.functional as F
preds_tensor = torch.cat(all_preds)
labels_tensor = torch.cat(all_labels)
accuracy = (preds_tensor == labels_tensor).float().mean().item()

import pandas as pd
import numpy as np
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/eval_logs"
os.makedirs(log_dir, exist_ok=True)
out_path = os.path.join(log_dir, f"ablation_{chunk_name}_{'_'.join(map(str, subset_global_indices))}_{timestamp}.csv")

df = pd.DataFrame({
    "prediction": preds_tensor.numpy(),
    "label": labels_tensor.numpy()
})
df.to_csv(out_path, index=False)

accuracy
