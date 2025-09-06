#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data/make_windows.py

TUAR → canonical 8ch windows (.npz shards) with a **patient-grouped, stratified split**.

Key features
------------
1) Robust TUAR parsing:
   - Canonicalize channels from TUAR montages (REF/LE/bipolar) to 8ch.
   - TUAR artifact CSV reader with combo-label normalization (eyem_musc → muscle, etc.).
   - Optional seizure CSVs.

2) Windowing + labels:
   - Fixed-length windows (sec) with overlap; bandpass + notch filters.
   - For each window, assign the SINGLE artifact label by overlapped time priority:
       electrode > muscle > chewing > eye > shiver > none
     Also compute intensity (overlap-weighted confidence ∈ [0,1]) and seizure flag.

3) **Two-pass** pipeline for good splits:
   - PASS A (fast): Pre-scan annotations (no signal) to compute per-file class histograms
     at your windowing granularity. Aggregate per-patient and run a greedy **group
     stratification** to fill train/val/test by target ratios while avoiding zero-class
     splits (with per-class minimums).
   - PASS B: Process signals only once, writing NPZ shards directly to their split.

4) Deterministic and configurable:
   - --seed for stable tie-breaks
   - --split_ratios
   - --min_val_per_class / --min_test_per_class (defaults ensure non-zero where feasible)

Artifacts (6-class)
-------------------
We assume (and emit) artifact classes in this fixed order:
["none","eye","muscle","chewing","shiver","electrode"]

Outputs per shard (.npz)
------------------------
x            : float32 [N, C=8, T]
y_artifact   : int64   [N] in [0..5]
y_seizure    : int64   [N] in {0,1}
y_agebin     : int64   [N] in {0,1,2,3}
y_montage    : int64   [N]
intensity    : float32 [N] ∈ [0,1]

Also writes meta.json with config and the computed split summary.
"""

import os, glob, argparse, json, math, random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
import mne

from utils.constants import CANON_CH, ARTIFACT_SET, age_to_bin_idx

# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------

def bandpass_filter(sig, fs, lo, hi):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, f0=60.0, Q=30.0):
    b, a = iirnotch(f0 / (fs / 2), Q)
    return filtfilt(b, a, sig)

# --------------------------------------------------------------------------------------
# TUAR montage → canonical 8ch
# --------------------------------------------------------------------------------------

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
    raw = raw.copy()
    if int(round(raw.info["sfreq"])) != fs_target:
        raw.resample(fs_target)
    name_map = {_norm_name(ch): ch for ch in raw.ch_names}
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

# --------------------------------------------------------------------------------------
# TUAR CSV (artifact + seizure)
# --------------------------------------------------------------------------------------

def _normalize_artifact_label(lbl: str) -> str:
    """
    Map TUAR labels (including combos) to canonical set:
      'eyem' -> 'eye', 'musc' -> 'muscle', 'chew' -> 'chewing',
      'shiv' -> 'shiver', 'elec' -> 'electrode', 'bckg' -> 'none'
    combos -> choose by priority: electrode > muscle > chewing > eye > shiver
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
        csv_path, sep=None, engine="python",
        comment="#", on_bad_lines="skip", skip_blank_lines=True,
    )
    cols = {c.lower().strip(): c for c in df.columns}
    start = df[cols.get("start_time", next(iter(cols)))]
    stop  = df[cols.get("stop_time", next(iter(cols)))]
    label = df[cols.get("label", next(iter(cols)))]
    conf  = df[cols["confidence"]] if "confidence" in cols else 1.0
    out = pd.DataFrame({
        "start_sec": np.asarray(start, dtype=float),
        "end_sec": np.asarray(stop, dtype=float),
        "artifact": [_normalize_artifact_label(x) for x in label],
        "confidence": (np.asarray(conf, dtype=float) if hasattr(conf, "__len__") else np.full(len(label), float(conf))),
    })
    out["start_sec"] = out["start_sec"].clip(0, rec_sec)
    out["end_sec"]   = out["end_sec"].clip(0, rec_sec)
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
        end   = np.asarray(df[cols["stop_time"]], dtype=float)
    elif "onset" in cols and "duration" in cols:
        start = np.asarray(df[cols["onset"]], dtype=float)
        end   = start + np.asarray(df[cols["duration"]], dtype=float)
    else:
        return pd.DataFrame(columns=["start_sec", "end_sec"])
    out = pd.DataFrame({"start_sec": start, "end_sec": end})
    out["start_sec"] = out["start_sec"].clip(0, rec_sec)
    out["end_sec"]   = out["end_sec"].clip(0, rec_sec)
    out = out[out["end_sec"] > out["start_sec"]]
    return out.reset_index(drop=True)

def _window_label_from_intervals(win_t0, win_t1, art_df: pd.DataFrame, seiz_df: pd.DataFrame):
    """
    Aggregate artifact annotations across time to a single window label by
    maximum overlapped seconds (priority is baked into normalization).
    Seizure flag if any overlap with seizure intervals.
    Intensity = overlap-weighted mean confidence in [0,1].
    """
    seiz = 0
    if len(seiz_df) > 0:
        ov = np.maximum(0.0, np.minimum(win_t1, seiz_df["end_sec"].values) - np.maximum(win_t0, seiz_df["start_sec"].values))
        if np.any(ov > 0):
            seiz = 1
    if len(art_df) == 0:
        return "none", seiz, 0.0
    start = art_df["start_sec"].values
    end   = art_df["end_sec"].values
    conf  = art_df["confidence"].values if "confidence" in art_df.columns else np.ones(len(art_df), dtype=float)
    ov = np.maximum(0.0, np.minimum(win_t1, end) - np.maximum(win_t0, start))
    mask = ov > 0
    if not np.any(mask):
        return "none", seiz, 0.0
    labels = art_df["artifact"].values
    totals, w_conf = {}, {}
    for o, lab, c in zip(ov[mask], labels[mask], conf[mask]):
        totals[lab] = totals.get(lab, 0.0) + float(o)
        w_conf[lab] = w_conf.get(lab, 0.0) + float(o * c)
    # Tie-break by priority if equal overlap
    prio = ["electrode", "muscle", "chewing", "eye", "shiver", "none"]
    best_lab, best_ov = None, -1.0
    for lab, tot in totals.items():
        if tot > best_ov:
            best_lab, best_ov = lab, tot
        elif abs(tot - best_ov) < 1e-6 and prio.index(lab) < prio.index(best_lab):
            best_lab = lab
    inten = float(w_conf.get(best_lab, 0.0) / (totals[best_lab] + 1e-6))
    inten = float(np.clip(inten, 0.0, 1.0))
    return best_lab, seiz, inten

# --------------------------------------------------------------------------------------
# Windowing (time-domain)
# --------------------------------------------------------------------------------------

def windowize_times(rec_sec: float, win_sec: float, overlap: float):
    """Return the list of window [t0,t1) in seconds for a recording."""
    step_sec = win_sec * (1.0 - overlap)
    if rec_sec < win_sec:
        return []
    n = int(math.floor((rec_sec - win_sec) / step_sec)) + 1
    return [(i * step_sec, i * step_sec + win_sec) for i in range(n)]

def windowize(X, fs, win_sec=4.0, overlap=0.5):
    step = int(win_sec * fs * (1 - overlap))
    W = int(win_sec * fs)
    starts = list(range(0, X.shape[1] - W + 1, step))
    out = np.stack([X[:, s:s + W] for s in starts], axis=0) if starts else np.zeros((0, X.shape[0], W), dtype=X.dtype)
    return out, step

# --------------------------------------------------------------------------------------
# Per-file processing (signals) — PASS B
# --------------------------------------------------------------------------------------

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
    rec_sec = raw.n_times / float(raw.info["sfreq"])
    art_df = _read_tuar_artifact_csv(csv_path, fs, rec_sec) if os.path.exists(csv_path) else pd.DataFrame(
        [{"start_sec": 0.0, "end_sec": rec_sec, "artifact": "none", "confidence": 1.0}]
    )
    seiz_df = _read_tuar_seiz_csv(edf_path, fs, rec_sec)

    # Map each window to (artifact, seizure, intensity)
    step_sec = win_sec * (1 - overlap)
    labels = []
    for i in range(W.shape[0]):
        t0 = float(i) * step_sec
        t1 = t0 + float(win_sec)
        art, seiz, inten = _window_label_from_intervals(t0, t1, art_df, seiz_df)
        labels.append((art, seiz, inten))

    y_artifact = np.array([ARTIFACT_SET.index(a) for a, _, _ in labels], dtype=np.int64)
    y_seizure  = np.array([s for _, s, _ in labels], dtype=np.int64)
    intensity  = np.clip(np.array([i for _, _, i in labels], dtype=np.float32), 0.0, 1.0)
    y_agebin   = np.full(W.shape[0], agebin, dtype=np.int64)
    y_montage  = np.full(W.shape[0], montage_id, dtype=np.int64)

    return W.astype(np.float32), y_artifact, y_seizure, y_agebin, y_montage, intensity

# --------------------------------------------------------------------------------------
# Writing shards
# --------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------
# PASS A: Pre-scan for stratified group split (patient-level)
# --------------------------------------------------------------------------------------

def patient_id_from_path(edf_path: str) -> str:
    """TUAR filename: edf/.../aaaaaaju_s005_t000.edf -> patient 'aaaaaaju'."""
    base = os.path.basename(edf_path)
    pid = base.split("_")[0]
    return pid

def csv_for(edf):  # expects CSV next to EDF, same stem + ".csv"
    c1 = os.path.splitext(edf)[0] + ".csv"
    return c1 if os.path.exists(c1) else ""

def file_class_histogram(edf_path: str, fs: int, win_sec: float, overlap: float) -> np.ndarray:
    """
    Compute per-file class counts by simulating window times from duration and
    assigning labels from artifact/seizure CSVs. Fast: no signal loading.
    """
    # read header only (preload=False) to get duration in seconds
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    rec_sec = raw.n_times / float(raw.info["sfreq"])
    times = windowize_times(rec_sec, win_sec, overlap)
    csvp = csv_for(edf_path)
    art_df = _read_tuar_artifact_csv(csvp, fs, rec_sec) if os.path.exists(csvp) else pd.DataFrame(
        [{"start_sec": 0.0, "end_sec": rec_sec, "artifact": "none", "confidence": 1.0}]
    )
    seiz_df = _read_tuar_seiz_csv(edf_path, fs, rec_sec)
    hist = np.zeros(len(ARTIFACT_SET), dtype=np.int64)
    for (t0, t1) in times:
        art, _, _ = _window_label_from_intervals(t0, t1, art_df, seiz_df)
        hist[ARTIFACT_SET.index(art)] += 1
    return hist  # per-file window counts by class

def greedy_group_stratify(edfs: list, file_hist: dict, split_ratios=(0.6,0.2,0.2),
                          min_val_per_class=1, min_test_per_class=1, seed=1234):
    """
    Group-aware greedy assignment of patients → {train,val,test} to match per-class ratios
    while avoiding 0-count classes in val/test. Returns file_to_split mapping and summary.

    Heuristic:
      - Aggregate per-patient class vectors G_p = sum over files.
      - Compute global totals T_k; desired split targets D_s,k = T_k * ratio_s.
      - Process patients in descending "rarity weight" order: dot(G_p, 1/T_k)
        so groups rich in rare classes are assigned first.
      - For each patient, pick the split s minimizing weighted L1 gap:
          cost_s = sum_k w_k * |(C_s,k + G_p,k) - D_s,k|
        where w_k = 1 / (T_k + 1e-9).
      - Tie-break by putting the group where the current count for its rarest present class is lowest.

    This simple greedy works well for imbalanced corpora like TUAR.
    """
    rng = random.Random(seed)

    # group by patient
    by_patient = defaultdict(list)
    for edf in edfs:
        pid = patient_id_from_path(edf)
        by_patient[pid].append(edf)

    # patient -> class vector
    patient_vec = {}
    for pid, files in tqdm(by_patient.items(), desc="Pre-scan: per-patient hist", leave=False):
        v = np.zeros(len(ARTIFACT_SET), dtype=np.int64)
        for f in files:
            v += file_hist[f]
        patient_vec[pid] = v

    # global totals
    T = np.zeros(len(ARTIFACT_SET), dtype=np.int64)
    for v in patient_vec.values():
        T += v
    ratios = list(split_ratios)
    assert abs(sum(ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
    D = [T * r for r in ratios]  # desired per-split per-class

    # weights (rare classes heavier)
    w = 1.0 / (T.astype(np.float64) + 1e-9)

    # order patients by rarity-weighted size
    order = sorted(patient_vec.keys(),
                   key=lambda p: float(np.dot(patient_vec[p], w)),
                   reverse=True)

    # current per-split tallies
    C = [np.zeros(len(ARTIFACT_SET), dtype=np.int64) for _ in range(3)]
    assign = {}  # pid -> split idx (0 train,1 val,2 test)

    for pid in tqdm(order, desc="Assign patients", leave=False):
        g = patient_vec[pid].astype(np.int64)
        # if a patient has zero windows (edge-case), push to train
        if g.sum() == 0:
            assign[pid] = 0
            continue

        # Choose split minimizing weighted L1 gap; tie-break with rarest-class need
        costs = []
        for s in range(3):
            cost = float(np.sum(w * np.abs((C[s].astype(np.float64) + g) - D[s])))
            costs.append((cost, s))
        costs.sort(key=lambda t: (t[0], t[1]))

        # prefer the split that is currently most lacking the rarest class present in this patient
        rare_present = np.where(g > 0)[0]
        # rarities by weight (higher w -> rarer)
        rare_present = sorted(rare_present, key=lambda k: w[k], reverse=True)

        best_s = costs[0][1]
        if rare_present:
            topk = rare_present[0]
            # among equal-cost splits, pick the one with smallest current count of that rare class
            tied_cost = costs[0][0]
            cand = [s for (c, s) in costs if abs(c - tied_cost) < 1e-9]
            if len(cand) > 1:
                best_s = sorted(cand, key=lambda s: C[s][topk])[0]

        assign[pid] = best_s
        C[best_s] += g

    # post-check: ensure min counts in val/test if feasible
    # If a class exists globally (T[k]>0) but C[1][k]==0 (val) or C[2][k]==0 (test),
    # try to flip one patient containing that class from another split.
    def try_fix(split_idx: int, needed_min: int):
        for k in range(len(ARTIFACT_SET)):
            if T[k] == 0: 
                continue
            if C[split_idx][k] >= needed_min:
                continue
            # find donor patient from other split with that class
            donor = None
            best_gain = 0
            for pid in order:
                s0 = assign[pid]
                if s0 == split_idx:
                    continue
                g = patient_vec[pid]
                if g[k] > 0:
                    # heuristic: prefer donors rich in class k and poor in others to reduce distortion
                    gain = g[k]
                    if gain > best_gain:
                        donor = pid
                        best_gain = gain
            if donor is not None:
                s0 = assign[donor]
                g = patient_vec[donor]
                assign[donor] = split_idx
                C[split_idx] += g
                C[s0] -= g

    # default minimums: at least 1 per class in val/test if globally present
    mv = max(0, int(min_val_per_class))
    mt = max(0, int(min_test_per_class))
    try_fix(1, mv)
    try_fix(2, mt)

    # build file->split map
    file_to_split = {}
    for pid, files in by_patient.items():
        s = assign[pid]
        for f in files:
            file_to_split[f] = s

    summary = {
        "global_class_totals": {ARTIFACT_SET[i]: int(T[i]) for i in range(len(T))},
        "split_ratios": ratios,
        "per_split_class_counts": [
            {ARTIFACT_SET[i]: int(C[s][i]) for i in range(len(T))}
            for s in range(3)
        ],
        "n_patients": len(by_patient),
        "patients_per_split": [
            int(sum(1 for p in assign if assign[p]==s)) for s in range(3)
        ]
    }
    return file_to_split, summary

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", required=True, help="path to TUAR root (contains edf/...)")
    ap.add_argument("--out_dir",   required=True, help="output directory for NPZ shards")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--win_sec", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0])
    ap.add_argument("--notch", type=float, default=60.0)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--split_ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help="train/val/test ratios; must sum to 1.0")
    ap.add_argument("--min_val_per_class", type=int, default=1,
                    help="ensure at least this many windows per class in VAL if feasible (0=disable)")
    ap.add_argument("--min_test_per_class", type=int, default=1,
                    help="ensure at least this many windows per class in TEST if feasible (0=disable)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard_size", type=int, default=4096)
    args = ap.parse_args()

    # 1) Collect EDFs
    edfs = sorted(glob.glob(os.path.join(args.tuar_root, "edf", "**/*.edf"), recursive=True))
    if not edfs:
        # also allow passing the edf directory directly
        edfs = sorted(glob.glob(os.path.join(args.tuar_root, "**/*.edf"), recursive=True))
    if not edfs:
        raise SystemExit(f"No EDFs found under {args.tuar_root}")

    # 2) PASS A: pre-scan per-file class histograms (from CSVs + durations only)
    print("[split] Pre-scan per-file histograms (annotations only, no signal)...")
    file_hist = {}
    for edf in tqdm(edfs, desc="Files", dynamic_ncols=True):
        try:
            file_hist[edf] = file_class_histogram(edf, fs=args.fs, win_sec=args.win_sec, overlap=args.overlap)
        except Exception as e:
            print(f"[warn] pre-scan failed for {edf}: {e}")
            file_hist[edf] = np.zeros(len(ARTIFACT_SET), dtype=np.int64)

    # 3) Patient-grouped greedy stratification
    print("[split] Greedy patient-group stratification...")
    file_to_split, split_summary = greedy_group_stratify(
        edfs, file_hist,
        split_ratios=tuple(args.split_ratios),
        min_val_per_class=args.min_val_per_class,
        min_test_per_class=args.min_test_per_class,
        seed=args.seed
    )

    # Make output dir structure
    os.makedirs(args.out_dir, exist_ok=True)

    # 4) PASS B: process signals and write shards per split
    split_names = ["train", "val", "test"]
    buckets = {0: [], 1: [], 2: []}  # list of (X,a,s,g,m,i) batches to be flushed in chunks
    counters = {0: 0, 1: 0, 2: 0}

    # Process files and stream to shards
    print("[build] Processing signals and writing shards...")
    for edf in tqdm(edfs, desc="Process", dynamic_ncols=True):
        sidx = file_to_split.get(edf, 0)  # default to train if missing
        csvp = csv_for(edf)
        X, a, s, g, m, i = process_record(
            edf, csvp, fs=args.fs, win_sec=args.win_sec,
            overlap=args.overlap, bp_lo=args.bandpass[0], bp_hi=args.bandpass[1],
            notch_f0=args.notch, montage_id=args.montage_id
        )
        if X.shape[0] == 0:
            print(f"[skip] no canonical channels or empty windows: {edf}")
            continue
        buckets[sidx].append((X, a, s, g, m, i))
        counters[sidx] += int(X.shape[0])

        # Flush shards eagerly to keep memory low
        if sum(len(b) for b in buckets[sidx]) * X.shape[0] >= args.shard_size * 2:
            out_split = os.path.join(args.out_dir, split_names[sidx])
            write_shards(buckets[sidx], out_split, split=split_names[sidx], shard_size=args.shard_size)
            buckets[sidx] = []

    # Final flush
    for sidx in (0,1,2):
        if buckets[sidx]:
            out_split = os.path.join(args.out_dir, split_names[sidx])
            write_shards(buckets[sidx], out_split, split=split_names[sidx], shard_size=args.shard_size)
            buckets[sidx] = []

    # 5) Write meta + split manifest
    meta = dict(
        fs=args.fs, win_sec=args.win_sec, overlap=args.overlap,
        bandpass=args.bandpass, notch=args.notch, montage_id=args.montage_id,
        split_ratios=args.split_ratios, canon_ch=CANON_CH,
        artifact_set=ARTIFACT_SET, seed=args.seed,
        stratified_split="patient-group greedy",
        split_summary=split_summary,
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Manifests: which EDFs ended up where
    man = defaultdict(list)
    for edf, s in file_to_split.items():
        man[split_names[s]].append(edf)
    for k,v in man.items():
        with open(os.path.join(args.out_dir, f"{k}_files.txt"), "w") as f:
            f.write("\n".join(sorted(v)))

    print("[done] Wrote shards and meta to", args.out_dir)
    print("        Split summary:", json.dumps(split_summary, indent=2))

if __name__ == "__main__":
    main()
