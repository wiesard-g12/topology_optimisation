"""
dataset.py - PyTorch Dataset & DataLoaders for Topology Optimization (Phase 2)

Supports:
- 6-channel input tensors (domain, bc_x, bc_y, Fx, Fy, Vf) and 1-channel target densities
- Automatic on-the-fly upgrade of legacy 5-channel datasets to 6 channels
- Strain energy density extraction (with fallback to targets proxy if not in .npz)
- Physics-consistent spatial data augmentations (Fx inverted under horizontal flip; Fy inverted under vertical flip)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TopoDataset(Dataset):
    """
    Dataset wrapper for SIMP topology optimization data.
    
    Args:
        data_path (str): Path to .npz file containing 'inputs', 'targets', and optionally 'strain_energy'
        augment (bool): Whether to apply physics-consistent spatial reflections
        return_strain_energy (bool): Whether to return strain energy tensor alongside inputs and targets
    """
    def __init__(self, data_path, augment=False, return_strain_energy=True):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")

        data = np.load(data_path)
        if 'inputs' not in data or 'targets' not in data:
            raise KeyError(f"Dataset {data_path} must contain 'inputs' and 'targets' keys.")

        inputs = data['inputs'].astype(np.float32)
        # Auto-upgrade 5-channel dataset to 6 channels if needed
        if inputs.ndim == 4 and inputs.shape[1] == 5:
            # [domain, bc, Fx, Fy, Vf] -> [domain, bc, bc, Fx, Fy, Vf]
            inputs = np.concatenate([
                inputs[:, 0:1],
                inputs[:, 1:2],
                inputs[:, 1:2],
                inputs[:, 2:5]
            ], axis=1)

        self.inputs = inputs
        self.targets = data['targets'].astype(np.float32)
        if self.targets.ndim == 3:
            self.targets = np.expand_dims(self.targets, axis=1)

        # Strain energy loading with graceful fallback
        if 'strain_energy' in data:
            self.strain_energy = data['strain_energy'].astype(np.float32)
            if self.strain_energy.ndim == 3:
                self.strain_energy = np.expand_dims(self.strain_energy, axis=1)
        else:
            # Fallback proxy: use targets as proxy so legacy .npz files continue to work
            self.strain_energy = self.targets.copy()

        self.augment = augment
        self.return_strain_energy = return_strain_energy
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.inputs[idx].copy()    # (6, 60, 120)
        y = self.targets[idx].copy()   # (1, 60, 120)
        se = self.strain_energy[idx].copy() # (1, 60, 120)

        if self.augment:
            # 1. Physics-consistent Horizontal Reflection (x -> -x, axis=2)
            # Invert Fx vector component (Channel 3). bc_x and bc_y are {0, 1} constraint masks.
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                x[3] = -x[3] # Invert Fx
                y = np.flip(y, axis=2).copy()
                se = np.flip(se, axis=2).copy()

            # 2. Physics-consistent Vertical Reflection (y -> -y, axis=1)
            # Invert Fy vector component (Channel 4).
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                x[4] = -x[4] # Invert Fy
                y = np.flip(y, axis=1).copy()
                se = np.flip(se, axis=1).copy()

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        se_tensor = torch.from_numpy(se)

        if self.return_strain_energy:
            return x_tensor, y_tensor, se_tensor
        return x_tensor, y_tensor


def get_dataloaders(data_path, batch_size=16, val_split=0.15, augment=True,
                    num_workers=0, seed=42, return_strain_energy=True):
    """
    Creates train and validation DataLoader instances with reproducible splitting.
    """
    full_dataset = TopoDataset(data_path, augment=False, return_strain_energy=return_strain_energy)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split)) if n_total > 1 else 0
    n_train = n_total - n_val

    if n_val > 0:
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)

        # Separate datasets so data augmentation is only applied to training set
        train_data = TopoDataset(data_path, augment=augment, return_strain_energy=return_strain_energy)
        val_data = TopoDataset(data_path, augment=False, return_strain_energy=return_strain_energy)

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
        train_data = TopoDataset(data_path, augment=augment, return_strain_energy=return_strain_energy)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = None

    return train_loader, val_loader
