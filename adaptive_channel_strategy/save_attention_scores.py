import os
import csv
from datetime import datetime

def save_attention_scores(scores, chunk_name, epoch, selected_channels, base_dir="attention_logs_clean"):
    """
    Save attention scores to a cleaned, organized directory structure.

    Args:
        scores (Tensor): Tensor of shape (B, Cin) with per-channel attention scores.
        chunk_name (str): Dataset chunk (e.g., "Allen", "HPA", "CP").
        epoch (int): Epoch number.
        selected_channels (List[int]): Global channel IDs used in this batch.
        base_dir (str): Root directory to save logs.
    """
    # Ensure directory structure: attention_logs_clean/{chunk}/{date}/
    date_str = datetime.now().strftime("%Y-%m-%d")
    chunk_dir = os.path.join(base_dir, chunk_name, date_str)
    os.makedirs(chunk_dir, exist_ok=True)

    # File name includes epoch and timestamp to avoid overwrite
    timestamp = datetime.now().strftime("%H-%M-%S")
    out_path = os.path.join(chunk_dir, f"attn_epoch{epoch}_{timestamp}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "chunk", "selected_channels"] + [f"channel_{i}" for i in range(scores.shape[1])]
        writer.writerow(header)
        for row in scores:
            writer.writerow([epoch, chunk_name, str(selected_channels)] + row.tolist())

    print(f"Saved attention scores to: {out_path}")
