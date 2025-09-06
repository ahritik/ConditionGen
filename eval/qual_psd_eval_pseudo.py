#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative & PSD eval-lite with OPTIONAL pseudo-labeling for real data.
- If real has y_artifact labels -> uses them.
- Else, if --pseudo is set -> score each real sample with band heuristics to build class-specific real pools.
- Else -> uses global real pool for all classes (not recommended).

Outputs per artifact:
  figs/qual_{artifact}.png         (real vs fake waveforms; per-trace z-scored for visibility)
  figs/psd_{artifact}.png          (PSD overlays, median ± IQR)
  metrics/metrics_psd_{artifact}.json (Δ bandpowers δ/θ/α/β)
And a table: summary_psd.md

Heuristic bands (Hz):
  drift:0.1-1  eye:0.5-3  chew:10-16  shiver:6-10  emg/muscle:20-45  theta:4-8  alpha:8-13  beta:13-30
"""
import os, glob, json, argparse, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import welch

# ---------- IO helpers ----------
def load_npz_any_key(path):
    with np.load(path, allow_pickle=True) as z:
        for k in ("x","X","signals","data","arr_0"):
            if k in z:
                arr = z[k].astype(np.float32); break
        else:
            keys = [k for k in z.files]
            if not keys: raise ValueError(f"No arrays in {path}")
            arr = z[keys[0]].astype(np.float32)
    if arr.ndim == 2: arr = arr[None,...]
    return arr  # (N,C,T)

def read_fs(meta_path, default_fs=200.0):
    try:
        j=json.load(open(meta_path))
        for k in ("fs","sfreq","sampling_rate","sample_rate"):
            if k in j: return float(j[k])
    except Exception:
        pass
    return float(default_fs)

def load_real(real_dir, max_files=600):
    paths = sorted(glob.glob(os.path.join(real_dir, "**", "*.npz"), recursive=True))
    if not paths: raise SystemExit(f"No .npz under {real_dir}")
    rng = np.random.default_rng(0)
    if len(paths) > max_files:
        paths = list(np.array(paths)[rng.choice(len(paths), max_files, replace=False)])
    buckets = defaultdict(list); have_labels=False
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            X=None
            for k in ("x","X","signals","data","arr_0"):
                if k in z: X=z[k].astype(np.float32); break
            if X is None: continue
            if X.ndim==2: X=X[None,...]
            y=z.get("y_artifact", None)
            if y is None:
                buckets["all"].append(X)
            else:
                have_labels=True
                y=np.array(y).reshape(-1)
                for aid in np.unique(y):
                    sel=X[y==aid]
                    if sel.size: buckets[int(aid)].append(sel)
    real={}
    for k,v in buckets.items():
        real[k]=np.concatenate(v,0)
    return real, have_labels

# ---------- signal features ----------
def welch_psd(x, fs, nperseg=1024):
    return welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)

def bandpower(x, fs, lo, hi):
    f,P = welch_psd(x, fs)
    m=(f>=lo)&(f<hi)
    return P[...,m].mean(axis=-1).mean()  # mean over freq then over channels

def bandpowers_4(x, fs):
    return np.array([
        bandpower(x, fs, 0.5, 4),
        bandpower(x, fs, 4, 8),
        bandpower(x, fs, 8, 13),
        bandpower(x, fs, 13, 30)
    ], dtype=np.float32)

def psd_curve(x, fs):
    f,P = welch_psd(x, fs)
    return f, P.mean(axis=0)

# ---------- pseudo-label scoring ----------
def score_artifacts(x, fs):
    """Return dictionary of simple scores for each artifact on a single (C,T) sample."""
    # powers
    p_drift = bandpower(x, fs, 0.1, 1.0)
    p_eye   = bandpower(x, fs, 0.5, 3.0)
    p_theta = bandpower(x, fs, 4, 8)
    p_alpha = bandpower(x, fs, 8, 13)
    p_chew  = bandpower(x, fs, 10, 16)
    p_shiv  = bandpower(x, fs, 6, 10)
    p_beta  = bandpower(x, fs, 13, 30)
    p_emg   = bandpower(x, fs, 20, 45)
    total   = p_drift + p_eye + p_theta + p_alpha + p_beta + p_emg + 1e-8
    # crude time-domain instability for electrode (drift/spikes)
    mean_trace = x.mean(axis=0)
    lowfreq_var = (mean_trace - mean_trace.mean()).std()
    # normalized ratios
    return {
        "eye":      p_eye / total,
        "muscle":   p_emg / total,
        "chewing":  p_chew / total,
        "shiver":   p_shiv / total,
        "electrode": 0.5*(p_drift/total) + 0.5*lowfreq_var/(np.sqrt((x**2).mean())+1e-6),
        "none":     1.0 - (p_eye+p_emg+p_chew+p_shiv+p_drift)/total
    }

def build_pseudo_real_pools(real_all, fs, top_k=2000, seed=0):
    """real_all: (N,C,T) unlabeled; returns dict name->(M,C,T)"""
    rng=np.random.default_rng(seed)
    N=real_all.shape[0]
    # For speed, subsample if huge
    idx = np.arange(N)
    if N>12000: idx = rng.choice(N, 12000, replace=False)
    X = real_all[idx]
    # score each sample
    scores = {k:[] for k in ["eye","muscle","chewing","shiver","electrode","none"]}
    for i in range(X.shape[0]):
        s = score_artifacts(X[i], fs)
        for k in scores: scores[k].append((s[k], i))
    # take top_k per class (or as many as available)
    pools={}
    for k in scores:
        ranked = sorted(scores[k], key=lambda t: t[0], reverse=True)
        take = ranked[:min(top_k, len(ranked))]
        sel = np.array([idx[j] for _, j in take], dtype=int)  # map back to original idx
        pools[k] = real_all[sel]
    return pools

# ---------- plotting ----------
def plot_qual(art, Xr, Xf, fs, out_png, n_examples=4, channels=(0,), seconds=5.0):
    seed = (abs(hash("qual_"+art)) % (2**32))
    rng = np.random.default_rng(seed)
    nR = min(len(Xr), n_examples); idxR = rng.choice(len(Xr), nR, replace=False)
    nF = min(len(Xf), n_examples); idxF = rng.choice(len(Xf), nF, replace=False)
    Xr = Xr[idxR]; Xf = Xf[idxF]
    C = Xr.shape[1]
    T = int(min(Xr.shape[-1], Xf.shape[-1], seconds*fs))
    t = np.arange(T)/fs
    rows = 2; cols = max(nR, nF)
    plt.figure(figsize=(1.8*cols, 1.5*rows*max(1,len(channels))))
    for j, X in enumerate([Xr, Xf]):
        for i in range(cols):
            if i >= len(X): continue
            ax = plt.subplot(rows, cols, j*cols+i+1)
            for ch in channels:
                ch_ = max(0, min(C-1, ch))
                s = X[i, ch_, :T]
                s = (s - s.mean()) / (s.std() + 1e-6)  # z-score per trace for visibility
                ax.plot(t, s, linewidth=0.9)
            ax.set_title(("real" if j==0 else "fake")+f" #{i}", fontsize=8)
            if i==0: ax.set_ylabel(art, fontsize=9)
            ax.set_xlabel("s"); ax.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_psd(art, Xr, Xf, fs, out_png):
    def curves(X):
        F=[]; A=[]
        for x in X:
            f,P = psd_curve(x, fs)
            F.append(f); A.append(P)
        return F[0], np.vstack(A)
    f, R = curves(Xr); _, F = curves(Xf)
    r_m = np.median(R,0); r_lo = np.percentile(R,25,0); r_hi = np.percentile(R,75,0)
    f_m = np.median(F,0); f_lo = np.percentile(F,25,0); f_hi = np.percentile(F,75,0)
    plt.figure(figsize=(5.0,3.0))
    plt.plot(f, r_m, label="real", linewidth=1.3)
    plt.fill_between(f, r_lo, r_hi, alpha=0.2)
    plt.plot(f, f_m, label="synthetic", linewidth=1.3)
    plt.fill_between(f, f_lo, f_hi, alpha=0.2)
    plt.xlim(0, 45); plt.xlabel("Hz"); plt.ylabel("PSD")
    plt.title(f"{art} — PSD (median ± IQR)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_parent", required=True)
    ap.add_argument("--label_map", default=None)  # optional; only used to list artifact names
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_real", type=int, default=3000)
    ap.add_argument("--n_fake", type=int, default=3000)
    ap.add_argument("--qual_channels", type=str, default="0")
    ap.add_argument("--qual_seconds", type=float, default=5.0)
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--pseudo", action="store_true", help="Use heuristic pseudo-labels for real if no labels present")
    ap.add_argument("--top_k", type=int, default=2000, help="Top-K real samples per class when pseudo-labeling")
    args = ap.parse_args()

    # sampling rate
    fs = args.fs if args.fs is not None else read_fs(os.path.join(args.real_dir,"meta.json"), 200.0)

    # artifact names
    arts = []
    if args.label_map and os.path.exists(args.label_map):
        try:
            lm = json.load(open(args.label_map))
            arts = lm.get("artifact_names") or lm.get("arts") or []
        except Exception:
            pass
    if not arts:
        arts = [os.path.basename(p).split("synth_")[-1] for p in sorted(glob.glob(os.path.join(args.fake_parent,"synth_*")))]
    if not arts:
        raise SystemExit("No artifacts found (no label_map and no synth_* folders).")

    out_dir = args.out_dir or os.path.join(args.fake_parent, "eval_figs_pseudo")
    figs_dir = os.path.join(out_dir, "figs"); os.makedirs(figs_dir, exist_ok=True)
    metrics_dir = os.path.join(out_dir, "metrics"); os.makedirs(metrics_dir, exist_ok=True)

    # load real
    real, have_labels = load_real(args.real_dir)
    # build unlabeled pool if needed
    real_global = None
    if not have_labels:
        real_global = real.get("all", None)
        if real_global is None and len(real)>0:
            real_global = np.concatenate([real[k] for k in real], 0)
        if real_global is None:
            raise SystemExit("No real data arrays found.")
    # prepare summary
    summary = ["# PSD Summary (Real vs Synthetic)\n",
               "| Artifact | Δδ | Δθ | Δα | Δβ | n_real | n_fake |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    chs = tuple(int(s) for s in args.qual_channels.split(","))

    # if unlabeled and pseudo requested, pre-build pools once
    pseudo_pools = {}
    if (not have_labels) and args.pseudo:
        print("Building pseudo-labeled real pools with heuristics...")
        pseudo_pools = build_pseudo_real_pools(real_global, fs, top_k=args.top_k)

    for art in arts:
        # pick real pool
        if have_labels:
            # try id by name (common 6-class)
            name2id = {"none":0,"eye":1,"muscle":2,"chewing":3,"shiver":4,"electrode":5}
            Xr = real.get(name2id.get(art, -999), None)
            if Xr is None:
                # fallback to global
                Xr = np.concatenate([real[k] for k in real if k!="all"], 0)
        else:
            if args.pseudo:
                Xr = pseudo_pools.get(art, None)
                if Xr is None:
                    print(f"[WARN] no pseudo pool for {art}; using global real")
                    Xr = real_global
            else:
                Xr = real_global
        if Xr is None:
            print(f"[WARN] skipping {art}: no real pool found"); continue
        # fake
        p_fake = os.path.join(args.fake_parent, f"synth_{art}", "samples.npy")
        if not os.path.exists(p_fake):
            print(f"[WARN] missing {p_fake}; skip {art}"); continue
        Xf = np.load(p_fake).astype(np.float32)
        if Xf.ndim==2: Xf = Xf[None,...]

        # subsample for speed
        rng=np.random.default_rng(abs(hash("sub_"+art)) % (2**32))
        if len(Xr) > args.n_real: Xr = Xr[rng.choice(len(Xr), args.n_real, replace=False)]
        if len(Xf) > args.n_fake: Xf = Xf[rng.choice(len(Xf), args.n_fake, replace=False)]

        # plots
        plot_qual(art, Xr, Xf, fs, os.path.join(figs_dir, f"qual_{art}.png"),
                  n_examples=args.examples, channels=chs, seconds=args.qual_seconds)
        plot_psd(art, Xr, Xf, fs, os.path.join(figs_dir, f"psd_{art}.png"))

        # metrics
        Rb = np.array([bandpowers_4(x, fs) for x in Xr]).mean(0)
        Fb = np.array([bandpowers_4(x, fs) for x in Xf]).mean(0)
        deltas = np.abs(Rb - Fb)
        m = {
            "artifact": art,
            "delta_delta": float(deltas[0]),
            "delta_theta": float(deltas[1]),
            "delta_alpha": float(deltas[2]),
            "delta_beta": float(deltas[3]),
            "n_real": int(len(Xr)),
            "n_fake": int(len(Xf))
        }
        json.dump(m, open(os.path.join(metrics_dir, f"metrics_psd_{art}.json"), "w"), indent=2)
        summary.append(f"| {art} | {m['delta_delta']:.3f} | {m['delta_theta']:.3f} | {m['delta_alpha']:.3f} | {m['delta_beta']:.3f} | {m['n_real']} | {m['n_fake']} |")

    with open(os.path.join(out_dir, "summary_psd.md"), "w") as f:
        f.write("\n".join(summary))
    print("\n".join(summary))
    print(f"\nSaved figs -> {figs_dir}")
    print(f"Saved metrics -> {metrics_dir}")

if __name__ == "__main__":
    main()
