# utils/constants.py
"""
Centralized constants & mappings for ConditionGen (TUAR ecosystem).
"""

from typing import List, Dict

# Canonical artifact set (7): none + 6 artifact types
ARTIFACT_SET: List[str] = ["none","eye","muscle","chewing","shiver","electrode","movement"]

# Canonical 8-channel montage
CANON_CH: List[str] = ["Fp1","Fp2","C3","C4","P3","P4","O1","O2"]

# Age bins (example): 0: child, 1: young adult, 2: adult, 3: senior
AGE_BINS = ["child","young","adult","senior"]

# Example montage ids you might encounter in TUAR/TUH
MONTAGE_IDS: Dict[str, int] = {
    "canon8": 0,  # our canonical 8-ch map
    "aasm20": 1,
    "tuh21": 2
}

# TUAR raw label to artifact map
# (Adjust keys as needed to match your TUAR CSV annotations)
TUAR_LABEL_MAP = {
    "eyem": "eye",
    "eye": "eye",
    "musc": "muscle",
    "chew": "chewing",
    "shiv": "shiver",
    "elec": "electrode",
    "move": "movement",
    "static": "electrode",
    "pop": "electrode",
    "none": "none",
}

def tuar_label_to_artifact(label: str) -> str:
    """Map TUAR raw event/annotation labels to our canonical artifact names."""
    if label is None:
        return "none"
    label = label.strip().lower()
    return TUAR_LABEL_MAP.get(label, "none")
