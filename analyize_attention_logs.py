import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime

attn_log_dir = "./attention_logs"
today_str = datetime.now().strftime("%Y-%m-%d")
target_hour = "21"
target_minute_range = range(44, 48)

attn_files = []
for f in os.listdir(attn_log_dir):
    if today_str in f and f.startswith("attn_epoch-1_unknown"):
        try:
            time_part = f.split("_")[-1].replace(".csv", "")  # '21-47-41'
            hour, minute, _ = time_part.split("-")
            if hour == target_hour and int(minute) in target_minute_range:
                attn_files.append(os.path.join(attn_log_dir, f))
        except Exception as e:
            print("Skipping file:", f, "due to error:", e)

attn_files.sort()
print("✅ Matching attention logs:", attn_files)

if attn_files:
    all_weights = []
    for f in attn_files:
        df = pd.read_csv(f)
        selected_channels = eval(df["selected_channels"].iloc[0])
        weights = df.iloc[0, 3:].values.astype(float)
        all_weights.append(weights)

    avg_weights = sum(all_weights) / len(all_weights)

    # Plot and save
    plt.figure(figsize=(10, 4))
    plt.bar([f"ch_{i}" for i in selected_channels], avg_weights)
    plt.title("Average Attention Weights Across Selected Channels")
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(attn_log_dir, f"avg_attention_plot_{today_str}_{target_hour}h.png")
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
else:
    print("⚠️ No matching attention log files found.")
