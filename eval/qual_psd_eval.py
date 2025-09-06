#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative & PSD eval-lite (no training).
- Loads real shards recursively from --real_dir (handles train/val/test).
- Loads synthetic from --fake_parent/synth_{artifact}/samples.npy
- Makes:
  figs/qual_{artifact}.png        (waveform grid)
  figs/psd_{artifact}.png         (PSD overlay with mean±IQR)
  metrics_psd_{artifact}.json     (bandpower deltas)
- Writes a summary table: summary_psd.md

Assumptions:
- Real NPZ has an array under key 'x' (fallback to any first array).
- Optional 'y_artifact' labels in real NPZ; if missing, uses global pool.
- Synthetic stored as samples.npy with shape (N,C,T).
"""
import os, glob, json, argparse, numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import welch

def read_fs(meta_path, default_fs):
    try:
        j = json.load(open(meta_path))
        for k in ("fs","sfreq","sampling_rate","sample_rate"):
            if k in j: return float(j[k])
    except Exception:
        pass
    return float(default_fs)

def load_npz_any_key(path):
    with np.load(path, allow_pickle=True) as z:
        for k in ("x","X","signals","data","arr_0"):
            if k in z:
                arr = z[k].astype(np.float32); break
        else:
            keys = [k for k in z.files]
            arr = z[keys[0]].astype(np.float32)
    if arr.ndim == 2: arr = arr[None,...]
    return arr  # (N,C,T), or (1,C,T)

def load_real(real_dir, max_files=500):
    paths = glob.glob(os.path.join(real_dir, "**", "*.npz"), recursive=True)
    paths = sorted(paths)
    if not paths: raise SystemExit(f"No .npz found under {real_dir}")
    rng = np.random.default_rng(0)
    if len(paths) > max_files:
        paths = list(np.array(paths)[rng.choice(len(paths), max_files, replace=False)])
    buckets = defaultdict(list)
    have_labels = False
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            X = None
            for k in ("x","X","signals","data","arr_0"):
                if k in z: X = z[k].astype(np.float32); break
            if X is None:
                ks = [k for k in z.files]
                if not ks: continue
                X = z[ks[0]].astype(np.float32)
            if X.ndim==2: X = X[None,...]
            y = z.get("y_artifact", None)
            if y is None:
                buckets["all"].append(X)
            else:
                have_labels = True
                y = np.array(y).reshape(-1)
                for aid in np.unique(y):
                    sel = X[y==aid]
                    if sel.size: buckets[int(aid)].append(sel)
    real = {}
    for k,v in buckets.items():
        real[k] = np.concatenate(v,0)
    return real, have_labels

def bandpowers_4(x, fs=200.0):
    # x: (C,T) -> 4-bandpower mean across channels
    f, P = welch(x, fs=fs, nperseg=min(1024, x.shape[-1]), axis=-1)
    def bp(lo,hi):
        m=(f>=lo)&(f<hi)
        return P[...,m].mean(axis=-1).mean()
    return np.array([bp(0.5,4), bp(4,8), bp(8,13), bp(13,30)], dtype=np.float32)

def psd_curve(x, fs=200.0):
    # x: (C,T) -> (F,) PSD averaged across channels
    f, P = welch(x, fs=fs, nperseg=min(1024, x.shape[-1]), axis=-1)
    return f, P.mean(axis=0)

def ensure_out(p):
    os.makedirs(p, exist_ok=True)
    return p

def name2id_default(name):
    table = {"none":0,"eye":1,"muscle":2,"chewing":3,"shiver":4,"electrode":5}
    return table.get(name, None)

def plot_qual(art, Xr, Xf, fs, out_png, n_examples=4, channels=(0,), seconds=5.0):
    rng = np.random.default_rng(0)
    nR = min(len(Xr), n_examples); idxR = rng.choice(len(Xr), nR, replace=False)
    nF = min(len(Xf), n_examples); idxF = rng.choice(len(Xf), nF, replace=False)
    Xr = Xr[idxR]; Xf = Xf[idxF]
    C = Xr.shape[1]
    T = int(min(Xr.shape[-1], Xf.shape[-1], seconds*fs))
    t = np.arange(T)/fs
    rows = 2; cols = max(nR, nF)
    plt.figure(figsize=(1.8*cols, 1.4*rows*len(channels)))
    for j, X in enumerate([Xr, Xf]):
        for i in range(cols):
            if i >= len(X): continue
            ax = plt.subplot(rows, cols, j*cols+i+1)
            for ch in channels:
                ch_ = max(0, min(C-1, ch))
                ax.plot(t, X[i, ch_, :T], linewidth=0.8)
            if j==0: ax.set_title(f"real #{i}", fontsize=8)
            else: ax.set_title(f"fake #{i}", fontsize=8)
            if i==0: ax.set_ylabel(art, fontsize=9)
            ax.set_xlabel("s"); ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_psd(art, Xr, Xf, fs, out_png):
    # compute per-sample curves averaged across channels
    r_curves=[]; f_curves=[]
    fgrid=None
    for x in Xr:
        f, P = psd_curve(x, fs)
        if fgrid is None: fgrid = f
        r_curves.append(P)
    for x in Xf:
        f, P = psd_curve(x, fs)
        f_curves.append(P)
    r_curves = np.vstack(r_curves); f_curves = np.vstack(f_curves)
    # mean ± IQR
    r_m = np.median(r_curves, axis=0); r_lo = np.percentile(r_curves,25,axis=0); r_hi = np.percentile(r_curves,75,axis=0)
    f_m = np.median(f_curves, axis=0); f_lo = np.percentile(f_curves,25,axis=0); f_hi = np.percentile(f_curves,75,axis=0)
    plt.figure(figsize=(5,3))
    plt.plot(fgrid, r_m, label="real", linewidth=1.3)
    plt.fill_between(fgrid, r_lo, r_hi, alpha=0.2)
    plt.plot(fgrid, f_m, label="synthetic", linewidth=1.3)
    plt.fill_between(fgrid, f_lo, f_hi, alpha=0.2)
    plt.xlim(0, 45); plt.xlabel("Hz"); plt.ylabel("PSD")
    plt.title(f"{art} — PSD (median ± IQR)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_parent", required=True)
    ap.add_argument("--label_map", default=None, help="JSON with artifact_names (optional)")
    ap.add_argument("--fs", type=float, default=None, help="sampling rate; if None, try real_dir/meta.json else 200")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_real", type=int, default=3000)
    ap.add_argument("--n_fake", type=int, default=3000)
    ap.add_argument("--qual_channels", type=str, default="0", help="comma sep channel idx e.g. '0' or '0,1'")
    ap.add_argument("--qual_seconds", type=float, default=5.0)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    fs = args.fs
    if fs is None:
        fs = read_fs(os.path.join(args.real_dir, "meta.json"), default_fs=200)

    arts = []
    if args.label_map and os.path.exists(args.label_map):
        try:
            lm = json.load(open(args.label_map))
            arts = lm.get("artifact_names") or lm.get("arts") or []
        except Exception:
            pass
    if not arts:
        # infer from synth_* folders
        arts = [os.path.basename(p).split("synth_")[-1] for p in sorted(glob.glob(os.path.join(args.fake_parent,"synth_*")))]
    if not arts:
        raise SystemExit("No artifacts found (no label_map and no synth_* folders).")

    out_dir = args.out_dir or os.path.join(args.fake_parent, "eval_figs")
    figs = ensure_out(os.path.join(out_dir, "figs"))
    metrics_dir = ensure_out(os.path.join(out_dir, "metrics"))
    summary_lines = ["# PSD Summary (Real vs Synthetic)\n",
                     "| Artifact | Δδ | Δθ | Δα | Δβ | n_real | n_fake |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
    # load real pools
    real, have_labels = load_real(args.real_dir)
    chs = tuple(int(s) for s in args.qual_channels.split(","))
    for art in arts:
        # gather real by label id if available
        if have_labels:
            aid = name2id_default(art)
            Xr = real.get(aid, None)
            if Xr is None:
                # fallback to concatenated pool
                Xr = np.concatenate([real[k] for k in real if k!="all"], 0) if real else None
        else:
            Xr = real.get("all", None)
        if Xr is None: 
            print(f"[WARN] no real found for {art}; skipping")
            continue
        # gather fake
        p = os.path.join(args.fake_parent, f"synth_{art}", "samples.npy")
        if not os.path.exists(p):
            print(f"[WARN] {p} missing; skipping {art}")
            continue
        Xf = np.load(p).astype(np.float32)
        if Xf.ndim==2: Xf = Xf[None,...]

        # subsample to keep things quick
        rng = np.random.default_rng(0)
        if len(Xr) > args.n_real:
            Xr = Xr[rng.choice(len(Xr), args.n_real, replace=False)]
        if len(Xf) > args.n_fake:
            Xf = Xf[rng.choice(len(Xf), args.n_fake, replace=False)]

        # qualitative
        plot_qual(art, Xr, Xf, fs, os.path.join(figs, f"qual_{art}.png"),
                  n_examples=args.examples, channels=chs, seconds=args.qual_seconds)

        # PSD overlays + metrics
        plot_psd(art, Xr, Xf, fs, os.path.join(figs, f"psd_{art}.png"))

        # band deltas
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
        summary_lines.append(f"| {art} | {m['delta_delta']:.3f} | {m['delta_theta']:.3f} | {m['delta_alpha']:.3f} | {m['delta_beta']:.3f} | {m['n_real']} | {m['n_fake']} |")

    with open(os.path.join(out_dir, "summary_psd.md"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print(f"\nSaved figures to: {figs}")
    print(f"Saved metrics to: {metrics_dir}")

if __name__ == "__main__":
    main()
