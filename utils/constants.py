#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/constants.py
------------------
Central constants for TUAR preprocessing and training.

- ARTIFACT_SET is the canonical 6-class taxonomy for TUAR (no "movement").
- CANON_CH defines the 8 canonical EEG channels we extract from varied montages.
- age_to_bin_idx() maps a numeric age to one of 4 discrete bins used in
  the 12-D conditioning vector (6 art one-hot + 1 seizure + 4 age one-hot + 1 montage).
"""

from __future__ import annotations
from typing import List

# -----------------------------------------------------------------------------
# Canonical TUAR artifact set (6 classes; NO "movement")
# Order here defines the integer ids written into NPZ shards and label_map.json
# -----------------------------------------------------------------------------
ARTIFACT_SET: List[str] = ["none", "eye", "muscle", "chewing", "shiver", "electrode"]

# -----------------------------------------------------------------------------
# Canonical 8-channel layout we target when canonicalizing TUAR montages
# (Fp1,Fp2,F3,F4,C3,C4,O1,O2). Keep case as shown; the converter normalizes.
# -----------------------------------------------------------------------------
CANON_CH: List[str] = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]

# -----------------------------------------------------------------------------
# Standard EEG bands used by eval/psd.py
# (Aligned with our preprocessing: bandpass 0.5–45 Hz)
# -----------------------------------------------------------------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# -----------------------------------------------------------------------------
# Age → 4-bin mapping used by the conditioning vector
# Adjust bins if your study pre-specifies different cut points; the model only
# sees 4 one-hot bins, not the raw age.
# -----------------------------------------------------------------------------
def age_to_bin_idx(age: int | float | None) -> int:
    """
    Map a (possibly missing) patient age to one of 4 discrete bins (0..3).

    Default bins:
      0: < 18 years
      1: 18–39
      2: 40–64
      3: 65+

    Returns: int in {0,1,2,3}
    """
    a = 40 if age is None else int(age)
    if a < 18:
        return 0
    elif a < 40:
        return 1
    elif a < 65:
        return 2
    else:
        return 3
