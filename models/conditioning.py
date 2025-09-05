#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conditioning.py
---------------
Single source of truth for ConditionGen’s conditioning schema.

Class taxonomy (TUAR):
    ARTIFACTS_CANON = ["none","eye","muscle","chewing","shiver","electrode"]  # 6 classes

Canonical conditioning vector (flat):
    12-D = 6(one-hot artifact) + 1(seizure) + 4(one-hot age) + 1(montage scalar)

This module also saves/loads a label_map.json so train/sample/eval share the
same class order. Keep using these helpers instead of hard-coding sizes.
"""

from __future__ import annotations
import os, json
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Canonical class order for TUAR: NO "movement"
# --------------------------------------------------------------------------------------
ARTIFACTS_CANON: List[str] = ["none", "eye", "muscle", "chewing", "shiver", "electrode"]

# ------------------------- Label-map helpers (save / load) ----------------------------

def save_label_map(dirpath: str, names: Optional[List[str]] = None) -> str:
    """
    Write the canonical artifact-name order next to your run logs/checkpoints.
    Downstream code (sampler, evaluator) reads this to align indices→names.
    """
    os.makedirs(dirpath, exist_ok=True)
    meta = {"artifact_names": names or ARTIFACTS_CANON}
    out = os.path.join(dirpath, "label_map.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    return out

def load_label_map_from(*paths: str, fallback: Optional[List[str]] = None) -> List[str]:
    """
    Given one or more paths (e.g., a checkpoint path or a data directory), search
    for a sibling/parent 'label_map.json' and return the artifact name list.
    If not found, return fallback or ARTIFACTS_CANON.
    """
    for p in paths:
        if not p:
            continue
        candidates = []
        if os.path.isdir(p):
            candidates.append(os.path.join(p, "label_map.json"))
        else:
            candidates.append(os.path.join(os.path.dirname(p), "label_map.json"))
            candidates.append(os.path.join(os.path.dirname(os.path.dirname(p)), "label_map.json"))
        for c in candidates:
            if os.path.exists(c):
                m = json.load(open(c))
                names = m.get("artifact_names")
                if isinstance(names, list) and names:
                    return names
    return (fallback or ARTIFACTS_CANON)

# ------------------------------- One-hot utilities -----------------------------------

def one_hot_np(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0
    return v

def one_hot_torch(i: int, n: int, device=None) -> torch.Tensor:
    v = torch.zeros(n, dtype=torch.float32, device=device); v[i] = 1.0
    return v

# --------------------------- Canonical 12-D builders ---------------------------------

def build_cond_np(
    artifact_idx: int,
    seizure: int,
    age_bin: int,
    montage_id: int,
    n_artifacts: int = 6,
) -> np.ndarray:
    """
    Build the flat 12-D conditioning vector (NumPy).
      6(one-hot artifact) + 1(seizure) + 4(one-hot age) + 1(montage scalar)
    """
    a = one_hot_np(artifact_idx, n_artifacts)                # [6]
    s = np.array([float(seizure)], dtype=np.float32)         # [1]
    g = one_hot_np(age_bin, 4)                               # [4]
    m = np.array([float(montage_id)], dtype=np.float32)      # [1]
    return np.concatenate([a, s, g, m], axis=0)              # -> (12,)

def build_cond_torch(
    artifact_idx: int,
    seizure: int,
    age_bin: int,
    montage_id: int,
    n_artifacts: int = 6,
    device=None,
) -> torch.Tensor:
    """
    PyTorch version of the 12-D builder (returns a 1-D tensor on 'device').
    """
    a = one_hot_torch(artifact_idx, n_artifacts, device)     # [6]
    s = torch.tensor([float(seizure)], dtype=torch.float32, device=device)  # [1]
    g = one_hot_torch(age_bin, 4, device)                    # [4]
    m = torch.tensor([float(montage_id)], dtype=torch.float32, device=device)  # [1]
    return torch.cat([a, s, g, m], dim=0)                    # -> (12,)

# ======================================================================================
# Legacy: condition embedding module (kept for compatibility with older code paths).
# If you pass the flat 12-D vector directly to UNet1DFiLM(cond_dim=12), you don't need this.
# ======================================================================================

class ConditionEmbed(nn.Module):
    """
    Legacy FiLM embedding block (not used by the new flat-vec path).
    Input concatenation is 12-D: 6(artifact one-hot) + 1(seizure) + 4(age one-hot) + 1(montage scalar)
    """
    def __init__(self, d_model: int = 128, n_montage: int = 8, use_intensity: bool = False):
        super().__init__()
        self.use_intensity = use_intensity

        # Minimal MLP over concatenated raw features (without montage embedding).
        self.mlp = nn.Sequential(
            nn.Linear(12, d_model),  # 6 + 1 + 4 + 1 = 12
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable embedding for montage_id if someone passes it as an index.
        self.montage_embed = nn.Embedding(n_montage, d_model)

        # Intensity was historically sampling-only; off by default.
        if use_intensity:
            self.intensity_proj = nn.Linear(1, d_model)

    def forward(
        self,
        artifact_onehot: torch.Tensor,  # [B,6]
        seizure: torch.Tensor,          # [B,1]
        age_onehot: torch.Tensor,       # [B,4]
        montage_id: torch.Tensor,       # [B,1] (index)
        intensity: Optional[torch.Tensor] = None,  # [B,1] or None
    ) -> torch.Tensor:
        base = torch.cat([artifact_onehot, seizure, age_onehot, montage_id], dim=-1)  # -> [B,12]
        h = self.mlp(base)
        h = h + self.montage_embed(montage_id.squeeze(-1).long())
        if self.use_intensity and (intensity is not None):
            h = h + self.intensity_proj(intensity)
        return h
