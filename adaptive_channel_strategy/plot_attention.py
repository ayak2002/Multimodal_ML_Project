import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# === Config ===
# Path to your cleaned attention logs
log_dir = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/attention_logs_clean/unknown/2025-04-10/attention_allen_nucleus_out"
log_pattern = os.path.join(log_dir, "attn_epoch-1_*.csv")

# Find matching files
attn_files = sorted(glob.glob(log_pattern))
print(f"🔍 Found {len(attn_files)} attention log files.")

if len(attn_files) == 0:
    print(" No attention logs found.")
    exit()

# Load and average
all_weights = []
for f in attn_files:
    df = pd.read_csv(f)
    scores = df.iloc[:, 3:].values  # Skip 'epoch', 'chunk', 'selected_channels'
    all_weights.append(scores)

# Stack and average
avg_weights = sum([w.mean(axis=0) for w in all_weights]) / len(all_weights)

# Plot
plt.figure(figsize=(8, 4))
plt.bar(range(len(avg_weights)), avg_weights)
plt.xticks(range(len(avg_weights)), [f"ch_{i}" for i in range(len(avg_weights))])
plt.xlabel("Channel")
plt.ylabel("Avg Attention Score")
plt.title("Average Attention Weights (allen subset no nucleus)")
plt.grid(True)
plt.tight_layout()

# Save figure
out_path = os.path.join(log_dir, "avg_attention_plot_allen_subset.png")
plt.savefig(out_path)
print(f"saved attention plot to {out_path}")