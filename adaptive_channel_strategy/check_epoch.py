import torch

#ckpt_path = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt"  # or full path if running elsewhere
ckpt_path = "/projectnb/cs598/projects/Modalities_Robustness/fission_strategy_ayak/multimodal_tests/checkpoints/morphem70k/2025-Apr-27-11-10-22--seed2025/model_last.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print(f"Epoch saved in checkpoint: {checkpoint.get('epoch', 'Not found')}")

# import torch

# # Path to your checkpoint
# ckpt_path = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt"

# Load the checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")

# List all top-level keys
print("Top-level keys:", checkpoint.keys())

# Check what's inside the model's state dict
model_keys = list(checkpoint["model_params"].keys())
print(f"Number of model parameters: {len(model_keys)}")
for k in model_keys:
    print(k)

# import torch

#ckpt_path = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/checkpoints/morphem70k/2025-Mar-27-14-31-39--seed2025/model_last.pt"
# ckpt_path = "/projectnb/cs598/projects/Modalities_Robustness/multimodal_tests/checkpoints/morphem70k/2025-Mar-14-10-33-06--seed2025/model_last.pt"
# ckpt = torch.load(ckpt_path, map_location='cpu')

# # List keys in the checkpoint
# print(ckpt.keys())
# for k in ckpt:
#     print(k, type(ckpt[k]))

# model_state = ckpt['model_params']

# for k in model_state:
#     if 'channel_attention' in k:
#         print(k, model_state[k].shape)
