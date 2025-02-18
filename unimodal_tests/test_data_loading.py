import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datasets.single_channel_dataset import SingleChannelDataset

def test_single_channel_loading():
    # Paths - using the correct location in channel_adaptive_models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root
    root_dir = os.path.join(base_dir, "channel_adaptive_models", "chammi_dataset", "CHAMMI")
    csv_path = os.path.join(os.path.dirname(root_dir), "morphem70k_v2.csv")
    
    print(f"Using dataset at: {root_dir}")
    print(f"Using CSV file: {csv_path}")
    
    # Test configurations
    test_cases = [
        {"modality": "Allen", "channel": "DNA"},
        {"modality": "HPA", "channel": "Nucleus"},
        {"modality": "CP", "channel": "Membrane"}
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['modality']} - {test_case['channel']} channel:")
        
        # Create dataset
        dataset = SingleChannelDataset(
            csv_path=csv_path,
            modality=test_case["modality"],
            channel_name=test_case["channel"],
            root_dir=root_dir,
            is_train=True
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        
        # Get a batch
        batch = next(iter(dataloader))
        images, labels = batch
        
        # Print information
        print(f"Number of samples in dataset: {len(dataset)}")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique labels: {torch.unique(labels).tolist()}")
        
        # Visualize first image in batch
        plt.figure(figsize=(5, 5))
        plt.imshow(images[0, 0].numpy(), cmap='gray')
        plt.title(f"{test_case['modality']} - {test_case['channel']}")
        save_dir = os.path.join(os.path.dirname(__file__), "visualization")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{test_case['modality']}_{test_case['channel']}.png"))
        plt.close()

def main():
    print("Starting data loading tests...")
    test_single_channel_loading()
    print("\nTests completed. Check the 'visualization' directory for sample images.")

if __name__ == "__main__":
    main() 