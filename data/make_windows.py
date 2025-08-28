# data/make_windows.py
"""
Create 4s windows @ 200Hz from TUAR EDFs + per-file CSV annotations.

This is a pragmatic script; adapt the paths/columns for your TUAR layout.
Requires: mne, pandas, numpy
"""
import os, argparse, json
import numpy as np
import pandas as pd
import mne
from utils.constants import CANON_CH, ARTIFACT_SET, tuar_label_to_artifact, MONTAGE_IDS

def find_edf_csv_pairs(root):
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".edf"):
                edf_path = os.path.join(dirpath, f)
                csv_path = os.path.splitext(edf_path)[0] + ".csv"
                if os.path.isfile(csv_path):
                    pairs.append((edf_path, csv_path))
    return pairs

def channel_map(raw):
    chs = {ch.upper(): i for i,ch in enumerate(raw.ch_names)}
    idxs = []
    mask = []
    for ch in CANON_CH:
        if ch.upper() in chs:
            idxs.append(chs[ch.upper()])
            mask.append(1.0)
        else:
            idxs.append(0)   # dummy index
            mask.append(0.0)
    return np.array(idxs), np.array(mask, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", type=str, required=True, help="Path to TUAR root")
    ap.add_argument("--out_npz", type=str, required=True, help="Output NPZ file (windows)")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--win_sec", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--bandpass", type=str, default="0.5,45.0")
    args = ap.parse_args()

    fs = args.fs
    win = int(args.win_sec * fs)
    hop = int(win * (1.0 - args.overlap))
    lo, hi = [float(x) for x in args.bandpass.split(",")]

    pairs = find_edf_csv_pairs(args.tuar_root)
    X, artifact, intensity, seizure, age_bin, montage_id = [], [], [], [], [], []

    for edf_path, csv_path in pairs:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"[WARN] EDF read failed {edf_path}: {e}")
            continue

        raw.resample(fs)
        raw.filter(lo, hi, verbose=False)

        idxs, mask_vec = channel_map(raw)
        data = raw.get_data(picks=idxs)  # (C, T)
        C, T = data.shape
        # z-score per channel
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)
        # apply mask
        data = data * mask_vec[:,None]

        # Load annotations (very dataset specific; here we expect start/stop secs + label)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] CSV read failed {csv_path}: {e}")
            continue

        # Expected columns; adapt as needed
        # start_time, stop_time, label (artifact), seizure (0/1), age (years or bin)
        if not set(["start_time","stop_time","label"]).issubset(df.columns):
            print(f"[WARN] CSV missing columns: {csv_path}")
            continue

        # Create a per-sample artifact timeline (naive: choose the strongest label overlapping the window center)
        # We'll map to canonical artifact index and a crude intensity [0..1] proportional to coverage.
        for s in range(0, T - win + 1, hop):
            e = s + win
            mid = (s + e) / 2.0 / fs  # seconds
            window = data[:, s:e]
            # default cond
            art_idx = ARTIFACT_SET.index("none")
            art_int = 0.0
            seiz = 0.0
            ageb = 2  # unknown->adult
            montage = MONTAGE_IDS["canon8"]

            overlapping = df[(df["start_time"] <= mid) & (df["stop_time"] >= mid)]
            if len(overlapping) > 0:
                # pick first row; or compute coverage-based intensity
                r = overlapping.iloc[0]
                art = tuar_label_to_artifact(str(r.get("label","none")))
                art_idx = ARTIFACT_SET.index(art)
                # crude intensity: fraction of window overlapped by annotated segment (if available)
                st = float(r.get("start_time", mid))
                et = float(r.get("stop_time", mid))
                cover = max(0.0, min(et, e/fs) - max(st, s/fs)) / (win/fs)
                art_int = float(np.clip(cover, 0.0, 1.0))
                seiz = float(r.get("seizure", 0.0))
                # map age to bin if available
                a = r.get("age", np.nan)
                if not pd.isna(a):
                    a = float(a)
                    if a < 13: ageb = 0
                    elif a < 25: ageb = 1
                    elif a < 60: ageb = 2
                    else: ageb = 3

            X.append(window.astype(np.float32))
            artifact.append(art_idx)
            intensity.append([art_int])
            seizure.append([seiz])
            age_bin.append(ageb)
            montage_id.append(montage)

    if len(X) == 0:
        print("No windows created. Check your TUAR paths and CSV schema.")
        return

    X = np.stack(X, axis=0)  # (N,C,T)
    artifact = np.array(artifact, dtype=np.int64)
    intensity = np.array(intensity, dtype=np.float32)
    seizure = np.array(seizure, dtype=np.float32)
    age_bin = np.array(age_bin, dtype=np.int64)
    montage_id = np.array(montage_id, dtype=np.int64)

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz,
        x=X,
        artifact=artifact,
        intensity=intensity,
        seizure=seizure,
        age_bin=age_bin,
        montage_id=montage_id
    )
    print(f"Wrote {args.out_npz} with {X.shape[0]} windows.")
if __name__ == "__main__":
    main()
