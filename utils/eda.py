#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/eda.py
------------
Exploratory Data Analysis for ConditionGen TUAR NPZ datasets.

Inputs
------
Base folder with:
  - label_map.json                # class names (from make_windows.py)
  - meta.json                     # fs, win_sec, etc. (optional)
  - {train,val,eval,test}/*.npz   # shards

Each NPZ shard may have NEW or LEGACY keys:
  - NEW: x, artifact, seizure, age_bin, montage_id, intensity
  - OLD: x, y_artifact, y_seizure, y_agebin, y_montage, intensity

Outputs
-------
Per split (under <base>/eda/<split>/):
  - artifact_counts.csv           (artifact, count, pct, duration_sec, pct_duration)
  - age_bin_counts.csv            (age_bin, count, pct)
  - montage_id_counts.csv         (montage_id, count)
  - channel_stats.csv             (channel, mean, std, rms)
  - intensity_hist.json           (present, bins, counts, stats)  # only if intensity exists

Per split (also in split dir itself):
  - <base>/<split>/summary.json   # split-level JSON summary

Top-level:
  - <base>/eda/summary_index.json # small index of the split summaries

Usage
-----
python utils/eda.py --base out/tuar_npz
"""

from __future__ import annotations
import os, glob, json, argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


FALLBACK_ARTIFACTS = ["none","eye","muscle","chewing","shiver","electrode"]


# ----------------------------- meta helpers ---------------------------------

def load_label_map(base_dir: str) -> List[str]:
    f = os.path.join(base_dir, "label_map.json")
    if os.path.exists(f):
        try:
            j = json.load(open(f))
            names = j.get("artifact_names")
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    return FALLBACK_ARTIFACTS

def load_meta(base_dir: str) -> dict:
    f = os.path.join(base_dir, "meta.json")
    if os.path.exists(f):
        try:
            return json.load(open(f))
        except Exception:
            pass
    return {}

def pick(z, *keys, default=None):
    for k in keys:
        if k in z: return z[k]
    return default


# ----------------------------- EDA core -------------------------------------

def analyze_split(split_dir: str,
                  class_names: List[str],
                  fs: float | None,
                  win_sec: float | None) -> dict:
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    split_name = os.path.basename(split_dir.rstrip("/"))
    if not files:
        return {"split": split_name, "n_files": 0, "n_windows": 0}

    name2idx = {n: i for i, n in enumerate(class_names)}
    n_cls = len(class_names)

    n_files = 0
    n_windows = 0
    C = None
    T = None

    # label tallies
    art_counts = np.zeros(n_cls, dtype=np.int64)
    age_counts = np.zeros(4, dtype=np.int64)
    seiz_count = 0
    montage_counts: Dict[int, int] = {}

    # intensity
    intensity_present = False
    inten_sum = 0.0
    inten_sumsq = 0.0
    inten_min = float("inf")
    inten_max = float("-inf")
    inten_n = 0
    # intensity histogram (20 bins in [0,1])
    bins = np.linspace(0.0, 1.0, 21)
    inten_hist = np.zeros(len(bins) - 1, dtype=np.int64)

    # channel stats accumulators
    ch_sum = None          # [C]
    ch_sumsq = None        # [C]
    ch_nsamp = 0           # total samples per-channel (= n_windows * T)

    for fp in files:
        with np.load(fp, allow_pickle=True) as z:
            X = z["x"]  # [N,C,T]
            N, c, t = X.shape
            if C is None:
                C, T = int(c), int(t)

            n_files += 1
            n_windows += int(N)

            A = pick(z, "artifact", "y_artifact")
            S = pick(z, "seizure", "y_seizure")
            G = pick(z, "age_bin", "y_agebin")
            M = pick(z, "montage_id", "y_montage")
            I = pick(z, "intensity")

            # artifacts → indices
            if A is None:
                raise KeyError(f"No artifact labels in {fp}")
            if getattr(A, "dtype", None) is not None and A.dtype.kind in ("U", "S", "O"):
                A_idx = np.array([name2idx[str(a)] for a in A], dtype=np.int64)
            else:
                A_idx = np.asarray(A, dtype=np.int64)

            # tallies
            art_counts += np.bincount(A_idx, minlength=n_cls)[:n_cls]
            if S is not None:
                seiz_count += int(np.asarray(S, dtype=np.int64).sum())
            if G is not None:
                age_counts += np.bincount(np.asarray(G, dtype=np.int64), minlength=4)[:4]
            if M is not None:
                for m in np.asarray(M, dtype=np.int64):
                    montage_counts[int(m)] = montage_counts.get(int(m), 0) + 1

            # intensity stats + hist
            if I is not None:
                intensity_present = True
                Ii = np.asarray(I, dtype=np.float64)
                inten_sum += float(Ii.sum())
                inten_sumsq += float((Ii * Ii).sum())
                inten_min = float(min(inten_min, float(Ii.min())))
                inten_max = float(max(inten_max, float(Ii.max())))
                inten_n += Ii.size
                h, _ = np.histogram(Ii, bins=bins)
                inten_hist += h

            # channel stats
            # accumulate sum over (N,T) for each channel
            s = X.sum(axis=(0, 2), dtype=np.float64)               # [C]
            s2 = np.square(X, dtype=np.float64).sum(axis=(0, 2))   # [C]
            if ch_sum is None:
                ch_sum = s
                ch_sumsq = s2
            else:
                ch_sum += s
                ch_sumsq += s2
            ch_nsamp += int(N * T)

    # finalize stats
    total = int(n_windows)
    total_sec = float(total) * float(win_sec) if win_sec else None
    total_hours = (total_sec / 3600.0) if total_sec is not None else None

    art_pct = (art_counts / max(1, total)).astype(float)
    durations = (art_counts * (win_sec if win_sec else 0.0)).astype(float)
    dur_pct = (durations / max(1.0, float(total) * (win_sec if win_sec else 1.0))).astype(float) if win_sec else np.zeros_like(art_pct)

    # channel means/std/rms
    ch_mean = (ch_sum / max(1, ch_nsamp)).astype(float).tolist()
    ch_var = (ch_sumsq / max(1, ch_nsamp) - (ch_sum / max(1, ch_nsamp)) ** 2).clip(min=0.0)
    ch_std = np.sqrt(ch_var).astype(float).tolist()
    ch_rms = np.sqrt((ch_sumsq / max(1, ch_nsamp))).astype(float).tolist()

    # intensity stats
    if intensity_present and inten_n > 0:
        inten_mean = inten_sum / inten_n
        inten_var = max(0.0, (inten_sumsq / inten_n) - inten_mean ** 2)
        inten_std = float(np.sqrt(inten_var))
        inten_stats = {
            "present": True,
            "n": int(inten_n),
            "mean": float(inten_mean),
            "std": float(inten_std),
            "min": float(inten_min),
            "max": float(inten_max),
        }
    else:
        inten_stats = {"present": False}

    # write CSVs
    eda_dir = os.path.join(os.path.dirname(split_dir), "eda", split_name)
    os.makedirs(eda_dir, exist_ok=True)

    # artifact counts table
    df_art = pd.DataFrame({
        "artifact": class_names,
        "count": art_counts.astype(int),
        "pct": art_pct,
        "duration_sec": durations,
        "pct_duration": dur_pct
    })
    df_art.to_csv(os.path.join(eda_dir, "artifact_counts.csv"), index=False)

    # age bin counts
    df_age = pd.DataFrame({
        "age_bin": [0, 1, 2, 3],
        "count": age_counts.astype(int),
        "pct": (age_counts / max(1, total)).astype(float)
    })
    df_age.to_csv(os.path.join(eda_dir, "age_bin_counts.csv"), index=False)

    # montage id counts
    mid_keys = sorted(montage_counts.keys())
    df_mont = pd.DataFrame({
        "montage_id": mid_keys,
        "count": [int(montage_counts[k]) for k in mid_keys]
    })
    df_mont.to_csv(os.path.join(eda_dir, "montage_id_counts.csv"), index=False)

    # channel stats
    df_ch = pd.DataFrame({
        "channel": list(range(C if C is not None else 0)),
        "mean": ch_mean,
        "std": ch_std,
        "rms": ch_rms
    })
    df_ch.to_csv(os.path.join(eda_dir, "channel_stats.csv"), index=False)

    # intensity histogram JSON (if present)
    if inten_stats.get("present", False):
        with open(os.path.join(eda_dir, "intensity_hist.json"), "w") as f:
            json.dump({
                "present": True,
                "bins": bins.tolist(),
                "counts": inten_hist.astype(int).tolist(),
                "stats": inten_stats
            }, f, indent=2)

    # split summary.json (ALSO placed inside split dir as requested)
    summary = {
        "split": split_name,
        "n_files": int(n_files),
        "n_windows": int(n_windows),
        "channels": int(C) if C is not None else None,
        "win_len": int(T) if T is not None else None,
        "fs": fs,
        "win_sec": win_sec,
        "total_sec": total_sec,
        "total_hours": total_hours,
        "seizure_windows": int(seiz_count),
        "seizure_rate": (seiz_count / max(1, n_windows)),
        "artifact_counts": {name: int(cnt) for name, cnt in zip(class_names, art_counts.tolist())},
        "artifact_pct": {name: float(p) for name, p in zip(class_names, art_pct.tolist())},
        "artifact_duration_sec": {name: float(d) for name, d in zip(class_names, durations.tolist())},
        "age_bin_counts": {str(i): int(v) for i, v in enumerate(age_counts.tolist())},
        "montage_id_counts": {str(k): int(v) for k, v in montage_counts.items()},
        "channel_stats": {"mean": ch_mean, "std": ch_std, "rms": ch_rms},
        "intensity": inten_stats
    }

    # write in eda/<split>/ and split/
    with open(os.path.join(eda_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(split_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base folder with train/val/eval/test NPZs")
    ap.add_argument("--splits", nargs="+", default=["train","val","eval","test"],
                    help="Which split subfolders to analyze (only existing ones will be processed)")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    class_names = load_label_map(base)
    meta = load_meta(base)
    fs = meta.get("fs", None)
    win_sec = meta.get("win_sec", None)

    os.makedirs(os.path.join(base, "eda"), exist_ok=True)
    index = {"base": base, "splits": []}

    for split in args.splits:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        print(f"[EDA] analyzing split: {split_dir}")
        summary = analyze_split(split_dir, class_names, fs, win_sec)
        out_path = os.path.join(base, "eda", split, "summary.json")
        index["splits"].append({"name": split, "summary": os.path.relpath(out_path, start=base)})

    with open(os.path.join(base, "eda", "summary_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[EDA] wrote index: {os.path.join(base, 'eda', 'summary_index.json')}")

if __name__ == "__main__":
    main()
