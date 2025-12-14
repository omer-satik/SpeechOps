# src/data_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SpectrogramDataset(Dataset):
    """
    Custom Dataset class that reads .npz files from data/processed directory.
    """
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        noisy_spec = data['noisy']
        clean_spec = data['clean']
        
        # Convert NumPy arrays to PyTorch tensors and add channel dimension [C, H, W]
        noisy_tensor = torch.from_numpy(noisy_spec).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_spec).float().unsqueeze(0)
        
        return noisy_tensor, clean_tensor
    
def pad_collate_fn(batch):
    """
    Takes a batch containing spectrograms of different lengths from DataLoader.
    Pads all spectrograms in the batch to match the longest one's size.
    """
    # Separate batch into noisy and clean spectrogram lists
    noisy_specs = [item[0] for item in batch]
    clean_specs = [item[1] for item in batch]

    # Find the time dimension (length) of the longest spectrogram in batch
    max_len = max([spec.shape[2] for spec in noisy_specs])

    # Pad noisy and clean spectrograms to the longest size
    # 'pad' function takes (left, right, top, bottom) padding amounts.
    # We only add padding to the right side (end of time dimension).
    padded_noisy = [F.pad(spec, (0, max_len - spec.shape[2]), "constant", 0) for spec in noisy_specs]
    padded_clean = [F.pad(spec, (0, max_len - spec.shape[2]), "constant", 0) for spec in clean_specs]

    # Now that all tensors are the same size, we can safely stack them into a single batch.
    final_noisy_batch = torch.stack(padded_noisy, 0)
    final_clean_batch = torch.stack(padded_clean, 0)

    return final_noisy_batch, final_clean_batch
