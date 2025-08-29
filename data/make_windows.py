import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
import mne

from utils.constants import CANON_CH, ARTIFACT_SET, age_to_bin_idx


# ------------------------------- Filters ------------------------------------ #

def bandpass_filter(sig, fs, lo, hi):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def notch_filter(sig, fs, f0=60.0, Q=30.0):
    b, a = iirnotch(f0 / (fs / 2), Q)
    return filtfilt(b, a, sig)


# --------------------- TUAR montage -> canonical 8ch ------------------------ #

def _norm_name(s: str) -> str:
    """
    Normalize raw channel names to a compact, uppercased form.
    Examples:
      'EEG FP1-REF' -> 'FP1-REF'
      'Fp1-F7'      -> 'FP1-F7'
      'C3'          -> 'C3'
    """
    s = (s or "").upper()
    s = s.replace("EEG ", "").replace(".", "")
    s = s.replace(" ", "").replace("__", "_")
    s = s.replace("--", "-").replace("_", "")
    return s


def _pick_one_target(raw, target: str, name_map):
    """
    Pick one best proxy channel for a canonical target (FP1, FP2, C3, ...).
    Preference:
      1) target-REF or target-LE (referential)            -> sign +1
      2) target-<neighbor>  (bipolar, target first)       -> sign +1
      3) <neighbor>-target  (bipolar, target second)      -> sign -1
    Returns: (original_channel_name, sign) or (None, +1)
    """
    # 1) Referential
    for suf in ("-REF", "-LE"):
        key = f"{target}{suf}"
        if key in name_map:
            return name_map[key], 1

    # 2) Bipolar with target first
    for k in name_map:
        if k.startswith(target + "-"):
            return name_map[k], 1

    # 3) Bipolar with target second (flip sign)
    for k in name_map:
        if k.endswith("-" + target):
            return name_map[k], -1

    return None, 1


def canonicalize(raw, fs_target=200):
    """
    Build an 8-channel canonical array from TUAR montages by selecting the best
    available channel per target and resampling. Accepts REF/LE or bipolar;
    flips sign if needed. Raises ValueError if no targets match.

    Returns:
      X   : float32 [C=8, T]
      mask: float32 [8] with 1 where a channel was found, 0 otherwise
    """
    # Work on a copy; ensure sampling first so lengths align
    raw = raw.copy()
    if int(round(raw.info["sfreq"])) != fs_target:
        raw.resample(fs_target)

    name_map = {_norm_name(ch): ch for ch in raw.ch_names}

    # CANON_CH from constants.py is ['Fp1','Fp2',...]; normalize to UPPER
    CANON_UP = [ch.upper() for ch in CANON_CH]
    T = raw.n_times
    X = np.zeros((len(CANON_UP), T), dtype=np.float32)
    mask = np.zeros(len(CANON_UP), dtype=np.float32)

    for i, tgt in enumerate(CANON_UP):
        orig, sgn = _pick_one_target(raw, tgt, name_map)
        if orig is None:
            continue
        x = raw.get_data(picks=[orig])[0].astype(np.float32)
        X[i] = sgn * x
        mask[i] = 1.0

    if mask.sum() == 0:
        raise ValueError("No canonical channel proxies found (TUAR montage not matched).")

    return X, mask


# -------------------------- TUAR CSV (robust reader) ------------------------ #

def _normalize_artifact_label(lbl: str) -> str:
    """
    Map TUAR labels (including combos) to canonical set:
      'eyem' -> 'eye', 'musc' -> 'muscle', 'chew' -> 'chewing',
      'shiv' -> 'shiver', 'elec' -> 'electrode', 'bckg' -> 'none'
      'eyem_musc' etc. -> choose by priority: electrode > muscle > chewing > eye > shiver
    """
    lbl = str(lbl or "").strip().lower()
    if lbl in {"bckg", "background", "none", "clean", ""}:
        return "none"
    parts = [p.strip() for p in lbl.split("_") if p.strip()]
    prio = ["elec", "musc", "chew", "eyem", "shiv"]
    alias = {
        "elec": "electrode",
        "musc": "muscle",
        "chew": "chewing",
        "eyem": "eye",
        "shiv": "shiver",
    }
    for p in prio:
        if p in parts:
            return alias[p]
    return alias.get(lbl, "none")


def _read_tuar_artifact_csv(csv_path: str, fs: int, rec_sec: float) -> pd.DataFrame:
    """
    Read TUAR artifact CSVs with header comments and variable delimiters.
    Returns DataFrame with columns: start_sec, end_sec, artifact, confidence
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["start_sec", "end_sec", "artifact", "confidence"])

    df = pd.read_csv(
        csv_path,
        sep=None,               # sniff delimiter
        engine="python",
        comment="#",            # ignore the header comment block
        on_bad_lines="skip",
        skip_blank_lines=True,
    )

    cols = {c.lower().strip(): c for c in df.columns}
    # TUAR canonical: channel,start_time,stop_time,label,confidence
    # be resilient to column variants
    start = df[cols.get("start_time", next(iter(cols)))]
    stop = df[cols.get("stop_time", next(iter(cols)))]
    label = df[cols.get("label", next(iter(cols)))]
    conf = df[cols["confidence"]] if "confidence" in cols else 1.0

    out = pd.DataFrame({
        "start_sec": np.asarray(start, dtype=float),
        "end_sec": np.asarray(stop, dtype=float),
        "artifact": [_normalize_artifact_label(x) for x in label],
        "confidence": (np.asarray(conf, dtype=float) if hasattr(conf, "__len__") else np.full(len(label), float(conf))),
    })
    # clip to record bounds and drop degenerate
    out["start_sec"] = out["start_sec"].clip(0, rec_sec)
    out["end_sec"] = out["end_sec"].clip(0, rec_sec)
    out = out[out["end_sec"] > out["start_sec"]]
    return out.reset_index(drop=True)


def _read_tuar_seiz_csv(edf_path: str, fs: int, rec_sec: float) -> pd.DataFrame:
    """
    If a seizure CSV exists (same dir, filename with _seiz.csv or .seiz.csv),
    return DataFrame with start_sec, end_sec. Otherwise empty.
    """
    base = os.path.splitext(edf_path)[0]
    candidates = [base + "_seiz.csv", base + ".seiz.csv"]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if csv_path is None:
        return pd.DataFrame(columns=["start_sec", "end_sec"])

    df = pd.read_csv(csv_path, sep=None, engine="python", comment="#", on_bad_lines="skip", skip_blank_lines=True)
    cols = {c.lower().strip(): c for c in df.columns}

    if "start_time" in cols and "stop_time" in cols:
        start = np.asarray(df[cols["start_time"]], dtype=float)
        end = np.asarray(df[cols["stop_time"]], dtype=float)
    elif "onset" in cols and "duration" in cols:
        start = np.asarray(df[cols["onset"]], dtype=float)
        end = start + np.asarray(df[cols["duration"]], dtype=float)
    else:
        return pd.DataFrame(columns=["start_sec", "end_sec"])

    out = pd.DataFrame({"start_sec": start, "end_sec": end})
    out["start_sec"] = out["start_sec"].clip(0, rec_sec)
    out["end_sec"] = out["end_sec"].clip(0, rec_sec)
    out = out[out["end_sec"] > out["start_sec"]]
    return out.reset_index(drop=True)


def _window_label_from_intervals(win_t0, win_t1, art_df: pd.DataFrame, seiz_df: pd.DataFrame):
    """
    Aggregate artifact annotations across time to a single window label by
    maximum overlapped seconds (priority is baked into normalization).
    Seizure flag if any overlap with seizure intervals.
    Intensity = overlap-weighted mean confidence in [0,1].
    """
    # seizure: any overlap
    seiz = 0
    if len(seiz_df) > 0:
        ov = np.maximum(0.0, np.minimum(win_t1, seiz_df["end_sec"].values) - np.maximum(win_t0, seiz_df["start_sec"].values))
        if np.any(ov > 0):
            seiz = 1

    if len(art_df) == 0:
        return "none", seiz, 0.0

    start = art_df["start_sec"].values
    end = art_df["end_sec"].values
    conf = art_df["confidence"].values if "confidence" in art_df.columns else np.ones(len(art_df), dtype=float)
    ov = np.maximum(0.0, np.minimum(win_t1, end) - np.maximum(win_t0, start))
    mask = ov > 0
    if not np.any(mask):
        return "none", seiz, 0.0

    labels = art_df["artifact"].values
    totals = {}
    w_conf = {}
    for o, lab, c in zip(ov[mask], labels[mask], conf[mask]):
        totals[lab] = totals.get(lab, 0.0) + float(o)
        w_conf[lab] = w_conf.get(lab, 0.0) + float(o * c)

    # Tie-break by priority if equal overlap
    prio = ["electrode", "muscle", "chewing", "eye", "shiver", "none"]
    best_lab = None
    best_ov = -1.0
    for lab, tot in totals.items():
        if tot > best_ov:
            best_lab, best_ov = lab, tot
        elif abs(tot - best_ov) < 1e-6:
            # tie -> use priority
            if prio.index(lab) < prio.index(best_lab):
                best_lab = lab

    inten = float(w_conf.get(best_lab, 0.0) / (totals[best_lab] + 1e-6))
    inten = float(np.clip(inten, 0.0, 1.0))
    return best_lab, seiz, inten


# ------------------------------- Windowing ---------------------------------- #

def windowize(X, fs, win_sec=4.0, overlap=0.5):
    step = int(win_sec * fs * (1 - overlap))
    W = int(win_sec * fs)
    starts = list(range(0, X.shape[1] - W + 1, step))
    out = np.stack([X[:, s:s + W] for s in starts], axis=0) if starts else np.zeros((0, X.shape[0], W), dtype=X.dtype)
    return out, step


def process_record(edf_path, csv_path, fs, win_sec, overlap, bp_lo, bp_hi, notch_f0, montage_id):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Basic patient age from EDF header if present
    age = raw.info.get("subject_info", {}).get("age", 40) or 40
    agebin = age_to_bin_idx(age)

    # Canonicalize to 8 channels (robust to TUAR montage)
    try:
        X, chmask = canonicalize(raw, fs_target=fs)
    except ValueError:
        # No suitable channels; signal empty
        return (
            np.zeros((0, 8, int(fs * win_sec)), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
        )

    # Filters (per-channel)
    if notch_f0 and notch_f0 > 0:
        X = np.vstack([notch_filter(X[c], fs, notch_f0)[None] for c in range(X.shape[0])])
    if bp_lo is not None and bp_hi is not None and bp_hi > bp_lo > 0:
        X = np.vstack([bandpass_filter(X[c], fs, bp_lo, bp_hi)[None] for c in range(X.shape[0])])

    # Z-score per channel (avoid zero-variance)
    for c in range(X.shape[0]):
        mu, sd = X[c].mean(), X[c].std() + 1e-6
        X[c] = (X[c] - mu) / sd

    # Windows
    W, step = windowize(X, fs, win_sec, overlap)  # [N,C,T]

    # TUAR annotations
    rec_sec = raw.times[-1]
    art_df = _read_tuar_artifact_csv(csv_path, fs, rec_sec) if os.path.exists(csv_path) else pd.DataFrame(
        [{"start_sec": 0.0, "end_sec": rec_sec, "artifact": "none", "confidence": 1.0}]
    )
    seiz_df = _read_tuar_seiz_csv(edf_path, fs, rec_sec)

    # Map each window to (artifact, seizure, intensity)
    win_starts = np.arange(W.shape[0]) * step
    labels = []
    for i in range(W.shape[0]):
        t0 = float(win_starts[i]) / fs
        t1 = t0 + float(win_sec)
        art, seiz, inten = _window_label_from_intervals(t0, t1, art_df, seiz_df)
        labels.append((art, seiz, inten))

    y_artifact = np.array([ARTIFACT_SET.index(a) for a, _, _ in labels], dtype=np.int64)
    y_seizure = np.array([s for _, s, _ in labels], dtype=np.int64)
    intensity = np.clip(np.array([i for _, _, i in labels], dtype=np.float32), 0.0, 1.0)
    y_agebin = np.full(W.shape[0], agebin, dtype=np.int64)
    y_montage = np.full(W.shape[0], montage_id, dtype=np.int64)

    return W.astype(np.float32), y_artifact, y_seizure, y_agebin, y_montage, intensity


def write_shards(items, out_dir, split="train", shard_size=4096):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    shard_id = 0
    Xs, A, S, G, M, I = [], [], [], [], [], []
    for X, a, s, g, m, i in items:
        for j in range(X.shape[0]):
            Xs.append(X[j]); A.append(a[j]); S.append(s[j]); G.append(g[j]); M.append(m[j]); I.append(i[j])
            idx += 1
            if idx % shard_size == 0:
                np.savez(
                    os.path.join(out_dir, f"{split}_{shard_id:03d}.npz"),
                    x=np.stack(Xs), y_artifact=np.array(A), y_seizure=np.array(S),
                    y_agebin=np.array(G), y_montage=np.array(M), intensity=np.array(I)
                )
                shard_id += 1
                Xs, A, S, G, M, I = [], [], [], [], [], []
    if Xs:
        np.savez(
            os.path.join(out_dir, f"{split}_{shard_id:03d}.npz"),
            x=np.stack(Xs), y_artifact=np.array(A), y_seizure=np.array(S),
            y_agebin=np.array(G), y_montage=np.array(M), intensity=np.array(I)
        )


# ---------------------------------- CLI ------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--win_sec", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0])
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--split_ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2])
    args = ap.parse_args()

    edfs = sorted(glob.glob(os.path.join(args.tuar_root, "**/*.edf"), recursive=True))
    if not edfs:
        raise SystemExit(f"No EDFs found under {args.tuar_root}")

    # Simple split by file
    n = len(edfs)
    n_train = int(n * args.split_ratios[0])
    n_val = int(n * args.split_ratios[1])
    train_files = edfs[:n_train]
    val_files = edfs[n_train:n_train + n_val]
    test_files = edfs[n_train + n_val:]

    def csv_for(edf):  # expects CSV next to EDF, same stem + ".csv"
        c1 = os.path.splitext(edf)[0] + ".csv"
        return c1 if os.path.exists(c1) else ""

    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        items = []
        for edf in tqdm(files, desc=f"Split {split}"):
            csvp = csv_for(edf)
            X, a, s, g, m, i = process_record(
                edf, csvp, fs=args.fs, win_sec=args.win_sec,
                overlap=args.overlap, bp_lo=args.bandpass[0], bp_hi=args.bandpass[1],
                notch_f0=args.notch, montage_id=args.montage_id
            )
            if X.shape[0] == 0:
                print(f"Skipping (no canonical channels): {edf}")
                continue
            items.append((X, a, s, g, m, i))
        if items:
            write_shards(items, args.out_dir, split=split)

    meta = dict(
        fs=args.fs, win_sec=args.win_sec, overlap=args.overlap,
        bandpass=args.bandpass, notch=args.notch, montage_id=args.montage_id,
        split_ratios=args.split_ratios, canon_ch=CANON_CH
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
