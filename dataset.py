"""
dataset.py - PyTorch Dataset and DataLoaders for Topology Optimization
Loads (N, 5, 60, 120) input tensors and (N, 60, 120) ground-truth density targets.
Includes physics-consistent spatial augmentation (vector-preserving reflections).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os


class TopoDataset(Dataset):
    """
    Dataset wrapper for SIMP topology optimization data.
    
    Args:
        data_path (str): Path to .npz file containing 'inputs' and 'targets'
        augment (bool): Whether to apply physics-consistent reflections
    """
    def __init__(self, data_path, augment=False):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
        data = np.load(data_path)
        if 'inputs' not in data or 'targets' not in data:
            raise KeyError(f"Dataset {data_path} must contain 'inputs' and 'targets' keys.")
            
        self.inputs = data['inputs'].astype(np.float32)    # (N, 5, 60, 120)
        self.targets = data['targets'].astype(np.float32)  # (N, 60, 120)
        
        # Ensure targets have channel dimension: (N, 1, 60, 120)
        if self.targets.ndim == 3:
            self.targets = np.expand_dims(self.targets, axis=1)
            
        self.augment = augment
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.inputs[idx].copy()    # (5, 60, 120)
        y = self.targets[idx].copy()   # (1, 60, 120)
        
        if self.augment:
            # 1. Physics-consistent Horizontal Reflection (x -> -x)
            # When domain is reflected horizontally, Fx vector component negates: Fx -> -Fx
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                x[2] = -x[2] # Invert Fx
                y = np.flip(y, axis=2).copy()
                
            # 2. Physics-consistent Vertical Reflection (y -> -y)
            # When domain is reflected vertically, Fy vector component negates: Fy -> -Fy
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                x[3] = -x[3] # Invert Fy
                y = np.flip(y, axis=1).copy()
                
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloaders(data_path, batch_size=16, val_split=0.15, augment=True, num_workers=0, seed=42):
    """
    Creates train and validation DataLoader instances.
    """
    full_dataset = TopoDataset(data_path, augment=False)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split)) if n_total > 1 else 0
    n_train = n_total - n_val
    
    if n_val > 0:
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
        
        # Separate datasets so augment is only applied to training set
        train_data = TopoDataset(data_path, augment=augment)
        val_data = TopoDataset(data_path, augment=False)
        
        train_subset = torch.utils.data.Subset(train_data, train_indices.indices)
        val_subset = torch.utils.data.Subset(val_data, val_indices.indices)
        
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=(len(train_subset) > batch_size)
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers
        )
    else:
        train_data = TopoDataset(data_path, augment=augment)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = None
        
    return train_loader, val_loader
