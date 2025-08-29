ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]
CANON_CH = ["Fp1","Fp2","C3","C4","P3","P4","O1","O2"]

ARTIFACT_MAP_TUAR = {
    "eyem":"eye", "eye":"eye",
    "musc":"muscle", "muscle":"muscle",
    "chew":"chewing", "chewing":"chewing",
    "shiv":"shiver", "shiver":"shiver",
    "elec":"electrode", "electrode":"electrode",
    "move":"movement", "movement":"movement",
    "none":"none", "clean":"none"
}

AGE_BINS = [(0,12),(13,25),(26,60),(61,200)]  # years; inclusive lower, inclusive upper

def tuar_label_to_artifact(s: str) -> str:
    s = (s or "").strip().lower()
    return ARTIFACT_MAP_TUAR.get(s, "none")

def age_to_bin_idx(age_years: float) -> int:
    for i,(lo,hi) in enumerate(AGE_BINS):
        if age_years >= lo and age_years <= hi:
            return i
    return 2  # default to adult

BANDS = {
    "delta": (0.5,4.0),
    "theta": (4.0,8.0),
    "alpha": (8.0,13.0),
    "beta":  (13.0,30.0),
}
