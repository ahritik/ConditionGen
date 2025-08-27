#!/usr/bin/env python3
import argparse, json, numpy as np
from scipy.signal import welch

BANDS = [(0.5,4),(4,8),(8,13),(13,30)]

def band_powers(x, fs):
    # x: (C,T)
    f, Pxx = welch(x, fs=fs, nperseg=fs*2, axis=-1)
    out = []
    for (a,b) in BANDS:
        m = (f>=a)&(f<b)
        out.append(Pxx[:,m].mean(axis=-1))
    return np.stack(out,0)  # (B,C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, required=True, help="NPZ with X,y,...")
    ap.add_argument("--gen", type=str, required=True, help="NPZ with X,...")
    ap.add_argument("--fs", type=int, default=200)
    args = ap.parse_args()

    zr = np.load(args.real, allow_pickle=True); Xr=zr["X"]; meta=zr["meta"].item() if isinstance(zr["meta"],str) else str(zr["meta"])
    zg = np.load(args.gen, allow_pickle=True); Xg=zg["X"]

    # sample same number
    n = min(len(Xr), len(Xg))
    Xr = Xr[:n]; Xg=Xg[:n]
    pr = np.stack([band_powers(x, args.fs) for x in Xr],0)  # (n,B,C)
    pg = np.stack([band_powers(x, args.fs) for x in Xg],0)

    rel_err = np.abs(pr - pg) / (np.abs(pr)+1e-8)
    out = {"rel_psd_err_mean": float(rel_err.mean()), "rel_psd_err_per_band": rel_err.mean(axis=(0,2)).tolist()}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
