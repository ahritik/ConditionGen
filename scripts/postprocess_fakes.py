#!/usr/bin/env python3
"""
Post-process generated windows to match training preprocessing:
Notch(60 Hz) -> Band-pass(0.5–45 Hz) -> per-window per-channel z-score
"""
import os, argparse, numpy as np
from glob import glob
from scipy.signal import iirnotch, butter, filtfilt

def notch60(x, fs, q=30):
    b,a = iirnotch(60.0/(fs/2), q)
    return filtfilt(b,a,x,axis=-1)

def bandpass(x, fs, lo=0.5, hi=45.0, order=4):
    b,a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b,a,x,axis=-1)

def per_channel_zscore(x, eps=1e-8):
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True)
    return (x - m) / (s + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Folder containing synth_* subdirs with samples.npy")
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=45.0)
    args = ap.parse_args()

    synth_dirs = sorted(glob(os.path.join(args.run_dir, "synth_*")))
    if not synth_dirs:
        print("[post] No synth_* dirs found under", args.run_dir); return

    for d in synth_dirs:
        p = os.path.join(d, "samples.npy")
        if not os.path.exists(p):
            print("[post] skip (no samples.npy):", d); continue
        X = np.load(p)   # [N,C,T]
        X = notch60(X, fs=args.fs)
        X = bandpass(X, fs=args.fs, lo=args.lo, hi=args.hi)
        X = per_channel_zscore(X)
        np.save(p, X.astype(np.float32))
        print(f"[post] wrote {p}  {X.shape} {X.dtype}")

if __name__ == "__main__":
    main()
