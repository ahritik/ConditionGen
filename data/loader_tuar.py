#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loader_tuar.py
--------------
NPZ shard loader for TUAR windows created by make_windows.py.

- Reads class order from label_map.json in the NPZ directory (6-class TUAR).
- Supports BOTH new shard keys:
      x, artifact, seizure, age_bin, montage_id, intensity
  and legacy keys:
      x, y_artifact, y_seizure, y_agebin, y_montage, intensity
- Builds the canonical 12-D conditioning vector:
      6(one-hot artifact) + 1(seizure) + 4(one-hot age) + 1(montage scalar)
"""

import os, glob, json
import numpy as np
import torch
from torch.utils.data import Dataset

from conditioning import build_cond_np, ARTIFACTS_CANON

def _load_label_map(npz_dir: str):
    lm = os.path.join(npz_dir, "label_map.json")
    if os.path.exists(lm):
        try:
            names = json.load(open(lm)).get("artifact_names")
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    return list(ARTIFACTS_CANON)

def _pick(z, *keys, default=None):
    for k in keys:
        if k in z: return z[k]
    if default is None: raise KeyError(f"None of the keys {keys} found")
    return default

class NPZShardDataset(Dataset):
    def __init__(self, npz_dir: str, split: str = "train",
                 shuffle_index: bool = True, as_tuple: bool = False):
        self.npz_dir = npz_dir
        self.split = split
        self.as_tuple = as_tuple

        self.files = sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No NPZ shards found in {npz_dir} for split={split}")

        self.class_names = _load_label_map(npz_dir)  # 6-class for TUAR
        self.n_artifacts = len(self.class_names)

        self.index = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = z["x"].shape[0]
            self.index.extend([(fi, i) for i in range(n)])

        if shuffle_index:
            np.random.default_rng(1234).shuffle(self.index)

        self._cache = None  # (fi, npz_obj)

    def __len__(self): return len(self.index)

    def _open(self, fi: int): return np.load(self.files[fi])

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]
        if self._cache is None or self._cache[0] != fi:
            if self._cache is not None:
                try: self._cache[1].close()
                except Exception: pass
            self._cache = (fi, self._open(fi))
        z = self._cache[1]

        x = z["x"][li].astype(np.float32)  # [C,T]

        art_arr = _pick(z, "artifact", "y_artifact")
        a = art_arr[li]
        if isinstance(a, (np.str_, str, bytes)):
            name2idx = {n: i for i, n in enumerate(self.class_names)}
            a = name2idx[str(a)]
        else:
            a = int(a)

        s = int(_pick(z, "seizure", "y_seizure")[li])
        g = int(_pick(z, "age_bin", "y_agebin")[li])
        m = int(_pick(z, "montage_id", "y_montage")[li])
        inten_arr = _pick(z, "intensity", default=None)
        inten = float(inten_arr[li]) if inten_arr is not None else 0.0

        cond_np = build_cond_np(a, s, g, m, n_artifacts=self.n_artifacts).astype(np.float32)

        x_t = torch.from_numpy(x).float()
        cond_t = torch.from_numpy(cond_np).float()

        if self.as_tuple:
            return x_t, cond_t

        return {
            "x": x_t,
            "artifact": a,
            "seizure": s,
            "age_bin": g,
            "montage_id": m,
            "intensity": inten,
            "cond_vec": cond_t,
        }
