import os, glob, json, numpy as np, torch
from torch.utils.data import Dataset
from utils.constants import ARTIFACT_SET

class NPZShardDataset(Dataset):
    """
    Loads NPZ shards produced by data/make_windows.py or scripts/make_mock_dataset.py

    Each .npz should contain:
      x: float32 [N, C=8, T]
      y_artifact: int64 [N] in [0..len(ARTIFACT_SET)-1]
      y_seizure: int64 [N] in {0,1}
      y_agebin: int64 [N] in {0,1,2,3}
      y_montage: int64 [N]
      intensity: float32 [N] in [0,1]
    """
    def __init__(self, npz_dir, split="train", shuffle_index=True):
        self.files = sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No NPZ shards found in {npz_dir} for split={split}")
        self.index = []  # (file_idx, local_idx)
        self.lengths = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = z["x"].shape[0]
            self.lengths.append(n)
            self.index.extend([(fi, i) for i in range(n)])
        if shuffle_index:
            rng = np.random.default_rng(1234)
            rng.shuffle(self.index)
        self.cache = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, fi):
        f = self.files[fi]
        return np.load(f)

    def __getitem__(self, idx):
        fi, li = self.index[idx]
        if self.cache is None or self.cache[0] != fi:
            self.cache = (fi, self._load_file(fi))
        z = self.cache[1]
        x = z["x"][li]          # [C,T]
        a = z["y_artifact"][li]
        s = z["y_seizure"][li]
        g = z["y_agebin"][li]
        m = z["y_montage"][li]
        inten = z["intensity"][li]
        cond = np.concatenate([
            np.eye(len(ARTIFACT_SET), dtype=np.float32)[a],
            np.array([float(s)], dtype=np.float32),
            np.eye(4, dtype=np.float32)[g],
            np.array([float(m)], dtype=np.float32)  # bare montage id; embedded later
        ], axis=0)
        sample = {
            "x": torch.from_numpy(x).float(),          # [C,T]
            "artifact": int(a),
            "seizure": int(s),
            "agebin": int(g),
            "montage": int(m),
            "intensity": float(inten),
            "cond_vec": torch.from_numpy(cond).float()
        }
        return sample
