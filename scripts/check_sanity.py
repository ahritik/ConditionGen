#!/usr/bin/env python3
"""
Sanity checks:
- print the first few condition vectors for each artifact
- compare per-channel std (mean over windows) real vs. fake
- save average PSD overlays (one figure)
"""
import os, argparse, json, numpy as np
from glob import glob
import matplotlib.pyplot as plt
from utils.constants import ARTIFACT_SET
from scipy.signal import welch

def load_real(npz_dir, split="train", maxn=5000):
    Xs = []
    fns = sorted(glob(os.path.join(npz_dir, split, "*.npz")))
    for p in fns:
        d = np.load(p)
        Xs.append(d["x"])  # [N,C,T], already preprocessed
        if sum(x.shape[0] for x in Xs) >= maxn: break
    X = np.concatenate(Xs, axis=0)
    return X

def load_fake(run_dir, artifact):
    p = os.path.join(run_dir, f"synth_{artifact}", "samples.npy")
    return np.load(p)

def cond_preview():
    from utils.constants import ARTIFACT_SET
    # mirror of sample.py cond
    def onehot(i, n):
        v = np.zeros(n, np.float32); v[i]=1; return v
    for a in ARTIFACT_SET[:3]:
        a_idx = ARTIFACT_SET.index(a)
        cond = np.concatenate([onehot(a_idx,7), [0.0], onehot(1,4), [0.0]], axis=0)
        print(f"[cond] {a}: len={len(cond)}  {cond}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--artifact", default="eye", choices=ARTIFACT_SET)
    ap.add_argument("--fs", type=float, default=200.0)
    args = ap.parse_args()

    cond_preview()

    Xr = load_real(args.real_dir, split="train", maxn=8000)     # [N,C,T]
    Xf = load_fake(args.run_dir, args.artifact)                  # [N,C,T]
    Cr = Xr.shape[1]; Cf = Xf.shape[1]
    assert Cr==Cf, "Channel count mismatch"

    # per-channel std (mean over windows)
    sr = Xr.std(axis=-1).mean(axis=0)
    sf = Xf.std(axis=-1).mean(axis=0)
    print("[std] real:", np.round(sr, 4))
    print("[std] fake:", np.round(sf, 4))

    # PSD overlay (avg over first 2048 windows)
    n = min(2048, Xr.shape[0], Xf.shape[0])
    fr, Pr = welch(Xr[:n].reshape(-1,Xr.shape[-1]), fs=args.fs, nperseg=256, axis=-1)
    ff, Pf = welch(Xf[:n].reshape(-1,Xf.shape[-1]), fs=args.fs, nperseg=256, axis=-1)
    Prm = Pr.mean(axis=0); Pfm = Pf.mean(axis=0)

    plt.figure()
    plt.loglog(fr, Prm, label="real")
    plt.loglog(ff, Pfm, label="fake")
    plt.xlabel("Hz"); plt.ylabel("PSD")
    plt.legend()
    outp = os.path.join(args.run_dir, f"psd_overlay_{args.artifact}.png")
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    print("[fig] wrote", outp)

if __name__ == "__main__":
    main()
