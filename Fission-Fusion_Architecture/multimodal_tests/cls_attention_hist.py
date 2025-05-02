import matplotlib.pyplot as plt
import torch
import numpy as np

# Dummy example — replace with actual preprocessed input
x = torch.randn(1, 12, 224, 224).to(device)  # Shape depends on your input
model = model.to(device)
_ = model(x, chunk_name="Allen", return_shared_specific=False)

def plot_cls_attention_histogram(model, device="cuda"):
    """
    Plot histogram of CLS token's attention scores across all heads and all layers.
    """
    all_cls_attns = []

    for layer_idx, blk in enumerate(model.feature_extractor.blocks):
        if hasattr(blk, "attn") and blk.attn.attn_map is not None:
            # attn_map shape: (batch, heads, tokens, tokens)
            attn_map = blk.attn.attn_map  # (B, num_heads, tokens, tokens)
            
            # Assume batch_size = 1 for simplicity (or you can loop)
            # CLS token is usually token 0
            cls_attn = attn_map[:, :, 0, :]  # (B, num_heads, tokens)
            cls_attn = cls_attn.reshape(-1)  # Flatten all heads and tokens
            
            all_cls_attns.append(cls_attn.cpu().numpy())

    all_cls_attns = np.concatenate(all_cls_attns)

    plt.figure(figsize=(8, 5))
    plt.hist(all_cls_attns, bins=100, density=True)
    plt.title("Histogram of CLS Attention Scores Across All Layers and Heads")
    plt.xlabel("Attention Score")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
