# src/data_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class SpectrogramDataset(Dataset):
    """
    data/processed klasöründeki .npz dosyalarını okuyan özel bir Dataset sınıfı.
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
        
        # NumPy dizilerini PyTorch tensörlerine dönüştür ve bir kanal boyutu ekle [C, H, W]
        noisy_tensor = torch.from_numpy(noisy_spec).float().unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_spec).float().unsqueeze(0)
        
        return noisy_tensor, clean_tensor
    
def pad_collate_fn(batch):
    """
    DataLoader'dan gelen ve farklı uzunluklardaki spektrogramları içeren bir 'batch'i alır.
    Batch'teki tüm spektrogramları, en uzun olanın boyutuna göre 'pad'ler (doldurur).
    """
    # Batch'i gürültülü ve temiz spektrogram listelerine ayır
    noisy_specs = [item[0] for item in batch]
    clean_specs = [item[1] for item in batch]

    # Batch'teki en uzun spektrogramın zaman boyutunu (uzunluğunu) bul
    max_len = max([spec.shape[2] for spec in noisy_specs])

    # Gürültülü ve temiz spektrogramları en uzun boyuta göre pad'le
    # 'pad' fonksiyonu (sol, sağ, üst, alt) dolgu miktarını alır.
    # Biz sadece sağ tarafa (zaman boyutunun sonuna) dolgu ekliyoruz.
    padded_noisy = [F.pad(spec, (0, max_len - spec.shape[2]), "constant", 0) for spec in noisy_specs]
    padded_clean = [F.pad(spec, (0, max_len - spec.shape[2]), "constant", 0) for spec in clean_specs]

    # Artık tüm tensörler aynı boyutta olduğuna göre, onları güvenle tek bir batch'te birleştirebiliriz.
    final_noisy_batch = torch.stack(padded_noisy, 0)
    final_clean_batch = torch.stack(padded_clean, 0)

    return final_noisy_batch, final_clean_batch
