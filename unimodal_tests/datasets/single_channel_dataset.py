import os
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
import skimage.io

class SingleChannelDataset(Dataset):
    """Dataset class for loading single channel data from CHAMMI dataset."""

    # Channel mappings for each modality
    CHANNEL_MAPS = {
        "Allen": {
            "DNA": 0,
            "Microtubules": 1,
            "Protein": 2
        },
        "HPA": {
            "Nucleus": 0,  
            "Microtubules": 1,
            "ER": 2,
            "Protein": 3
        },
        "CP": {
            "DNA": 0,      
            "RNA": 1,      
            "ER": 2,       
            "Actin": 3,    
            "Membrane": 4 
        }
    }

    # Class mappings for each modality (same as original implementation)
    CLASSES = {
        "Allen": {
            "M0": 0,
            "M1M2": 1,
            "M3": 2,
            "M4M5": 3,
            "M6M7_complete": 4,
            "M6M7_single": 5,
        },
        "HPA": {
            "golgi apparatus": 0,
            "microtubules": 1,
            "mitochondria": 2,
            "nuclear speckles": 3,
        },
        "CP": {
            "BRD-A29260609": 0,
            "BRD-K04185004": 1,
            "BRD-K21680192": 2,
            "DMSO": 3,
        }
    }

    def __init__(
        self,
        csv_path: str,
        modality: str,  # "Allen", "HPA", or "CP"
        channel_name: str,  # e.g., "DNA", "Membrane", etc.
        root_dir: str,
        is_train: bool,
        transform: Optional[Callable] = None,
        target_labels: str = "label"
    ):
        """
        Initialize the single channel dataset.
        
        Args:
            csv_path: Path to the metadata CSV file
            modality: Which modality to use (Allen, HPA, CP)
            channel_name: Which channel to extract
            root_dir: Directory with all the images
            is_train: True for training set
            transform: Optional transforms to apply
            target_labels: Column name for labels in CSV
        """
        assert modality in self.CHANNEL_MAPS, f"Modality must be one of {list(self.CHANNEL_MAPS.keys())}"
        assert channel_name in self.CHANNEL_MAPS[modality], (
            f"Channel {channel_name} not found in modality {modality}. "
            f"Available channels: {list(self.CHANNEL_MAPS[modality].keys())}"
        )

        self.modality = modality
        self.channel_name = channel_name
        self.channel_idx = self.CHANNEL_MAPS[modality][channel_name]
        self.is_train = is_train
        self.transform = transform
        self.root_dir = root_dir
        
        # Read and filter metadata
        self.metadata = pd.read_csv(csv_path)
        self.metadata = self.metadata[self.metadata["chunk"] == modality]
        if is_train:
            self.metadata = self.metadata[self.metadata["train_test_split"] == "Train"]
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Set up class mapping
        self.classes_dict = self.CLASSES[modality]
        self.target_labels = target_labels

    def __len__(self):
        return len(self.metadata)

    def _fold_channels(self, image: np.ndarray, channel_width: int) -> Tensor:
        """
        Convert image from tape format to tensor, extracting only the specified channel.
        
        Args:
            image: Input image in tape format (h, w * c)
            channel_width: Width of each channel
            
        Returns:
            Tensor of shape (1, h, w) containing only the specified channel
        """
        # First reshape to (h, w, c)
        output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
        
        # Extract only the channel we want and add channel dimension
        output = output[:, :, self.channel_idx]
        output = output[np.newaxis, :, :]  # Shape: (1, h, w)
        
        # Convert to tensor and normalize to [0, 1]
        output = torch.from_numpy(output).float()
        output = output / 255.0
        
        return output

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple of (image, label) where image is a single-channel tensor
            of shape (1, h, w) and label is the class index
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join(self.root_dir, self.metadata.loc[idx, "file_path"])
        channel_width = self.metadata.loc[idx, "channel_width"]
        image = skimage.io.imread(img_path)
        
        # Convert to single channel tensor
        image = self._fold_channels(image, channel_width)

        # Get label if in training mode
        if self.is_train:
            label = self.metadata.loc[idx, self.target_labels]
            label = self.classes_dict[label]
            label = torch.tensor(label)
        else:
            label = None

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, label
        else:
            return image 