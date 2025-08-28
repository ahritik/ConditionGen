# data/loaders_tuar_tusz.py
"""
Windowed EEG dataset loader for TUAR ecosystem.
Assumes make_windows.py created NPZ files with arrays:
  x: (N, C, T)
  artifact: (N,) int
  intensity: (N,1) float
  seizure: (N,1) float
  age_bin: (N,) int
  montage_id: (N,) int
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class WindowedEEGDataset(Dataset):
    def __init__(self, npz_path):
        assert os.path.isfile(npz_path), f"NPZ not found: {npz_path}"
        self.data = np.load(npz_path)
        self.x = self.data["x"]          # (N,C,T)
        self.artifact = self.data["artifact"]
        self.intensity = self.data["intensity"]
        self.seizure = self.data["seizure"]
        self.age_bin = self.data["age_bin"]
        self.montage_id = self.data["montage_id"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        cd = {
            "artifact": torch.tensor(self.artifact[idx], dtype=torch.long),
            "intensity": torch.tensor(self.intensity[idx:idx+1], dtype=torch.float32).view(1,-1),  # (1,1) -> will concat in model
            "seizure": torch.tensor(self.seizure[idx:idx+1], dtype=torch.float32).view(1,-1),
            "age_bin": torch.tensor(self.age_bin[idx], dtype=torch.long),
            "montage_id": torch.tensor(self.montage_id[idx], dtype=torch.long),
        }
        # flatten (B,1) to (B,1) later; keep shapes consistent in training loop
        return x, cd

def make_loader(npz_path, batch_size=32, shuffle=True, num_workers=2, pin_memory=False):
    ds = WindowedEEGDataset(npz_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
