import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the directory containing attention logs
log_dir = "/projectnb/cs598/projects/Modalities_Robustness/adaptive_channel_strategy/attention_logs_clean/unknown/2025-04-10/attention_allen_nucleus_out"

# List all CSV files in the directory
files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
print(f"Found {len(files)} attention log files.")

# Load and average attention scores across files
all_scores = []
for file in files:
    df = pd.read_csv(os.path.join(log_dir, file))
    # Only include numeric columns
    numeric_cols = df.select_dtypes(include=np.number)
    scores = numeric_cols.mean(axis=0).values
    all_scores.append(scores)

# Compute mean and standard deviation
mean_scores = np.mean(all_scores, axis=0)
std_scores = np.std(all_scores, axis=0)

# Plot the results
x = np.arange(len(mean_scores))
plt.figure(figsize=(10, 4))
plt.bar(x, mean_scores, yerr=std_scores, capsize=5)
plt.xlabel("Allen Channel Index")
plt.ylabel("Average Attention Score")
plt.title("Attention Weights - Allen (Nucleus Left Out)")
plt.axvline(x=0, color="red", linestyle="--", label="Nucleus (Left Out)")
plt.legend()
plt.tight_layout()
plt.savefig("fig1b_allen_nucleus_left_out.png")
plt.show()
