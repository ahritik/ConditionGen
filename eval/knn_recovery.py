#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/knn_recovery.py — fast baseline+recovery with k-NN (no neural training)

New in this version:
- --feature bp_stat_spatial : bandpowers per band aggregated as (mean,std) across channels
  + low-freq (0.5–2) for EOG, + high-freq (30–45) for EMG
- optional spectral slope (1/f) features
- standardization and class-balanced train bank help avoid single-class collapse
"""

from __future__ import annotations
import os, glob, json, argparse
from typing import Dict, Tuple, List, Optional
import numpy as np

# 6-class fallback
CLASSES = ["none","eye","muscle","chewing","shiver","electrode"]
# Bands for spatial features:
BANDS_SPATIAL = {
  "lf":   (0.5, 2.0),   # EOG-ish
  "theta":(4.0, 8.0),
  "alpha":(8.0, 13.0),
  "beta": (13.0, 30.0),
  "hf":   (30.0, 45.0), # EMG-ish
}
# Legacy bands for bp_only / bp_stat (global average)
BANDS_LEGACY = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--augment_artifact", default=None)

    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--feature", choices=["bp_stat","bp_only","stat_only","bp_stat_spatial"], default="bp_stat_spatial",
                    help="bp_stat_spatial recommended; others kept for compatibility")
    ap.add_argument("--add_spec_slope", action="store_true", help="append spectral slope+fit R^2 (global)")

    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit_train_per_class", type=int, default=3000)
    ap.add_argument("--limit_test", type=int, default=10000)
    ap.add_argument("--limit_fake", type=int, default=8000)
    ap.add_argument("--standardize", action="store_true", help="z-score features using train mean/std")

    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--metric", choices=["cosine","euclidean"], default="euclidean")
    ap.add_argument("--weighted", action="store_true", help="weighted vote")

    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def tqdmit(it, enabled, desc):
    if not enabled: return it
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, dynamic_ncols=True, leave=False)
    except Exception:
        return it

def load_label_map(base: str) -> List[str]:
    p = os.path.join(base, "label_map.json")
    if os.path.exists(p):
        try:
            j = json.load(open(p))
            names = j.get("artifact_names")
            if isinstance(names, list) and names: return names
        except Exception: pass
    return CLASSES

def list_npz(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.npz")))

def windows_from_dir(folder: str, label_key="y_artifact", stride=1, limit=0, use_tqdm=False):
    files = list_npz(folder)
    xs, ys = [], []
    it = tqdmit(files, use_tqdm, f"load {os.path.basename(folder)}")
    n_acc = 0
    for f in it:
        with np.load(f, allow_pickle=True) as z:
            X = z["x"][::max(1,stride)]
            if label_key in z:
                Y = z[label_key][::max(1,stride)]
            elif "artifact" in z:
                Y = z["artifact"][::max(1,stride)]
            else:
                Y = z["y_artifact"][::max(1,stride)]
        xs.append(X); ys.append(Y)
        n_acc += len(Y)
        if limit and n_acc >= limit: break
    if not xs:
        return np.empty((0,8,1), np.float32), np.empty((0,), np.int64)
    X = np.concatenate(xs, 0).astype(np.float32)
    Y = np.concatenate(ys, 0).astype(np.int64)
    if limit and len(Y) > limit:
        X, Y = X[:limit], Y[:limit]
    return X, Y

def per_class_limits(Y: np.ndarray, limit_per_class: int, n_cls: int) -> np.ndarray:
    if not limit_per_class or limit_per_class<=0: return np.arange(len(Y))
    keep=[]; cnt=np.zeros(n_cls, dtype=int)
    for i,y in enumerate(Y):
        if 0 <= y < n_cls and cnt[y] < limit_per_class:
            keep.append(i); cnt[y]+=1
    return np.array(keep, dtype=int)

# ---------- FFT helpers ----------
def psd_per_channel(x: np.ndarray, fs: int):
    """x:[C,T] -> (f:[F], psd:[C,F]), fast single-window Welch."""
    C,T=x.shape
    win = np.hanning(T).astype(np.float32)
    f = np.fft.rfftfreq(T, 1.0/fs)
    P=[]
    for c in range(C):
        Xw=np.fft.rfft(x[c]*win)
        P.append((Xw.real**2 + Xw.imag**2) / (np.sum(win**2)*fs))
    return f, np.stack(P,0)  # [C,F]

def bandpower_chan(psd_c: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    m = (f>=lo)&(f<hi)
    # integrate power per channel over the band
    bp = np.trapezoid(psd_c[:, m], f[m], axis=1)  # [C]
    # normalize by total band 0.5-45 to reduce amplitude bias
    m_tot = (f>=0.5)&(f<45.0)
    tot = np.trapezoid(psd_c[:, m_tot], f[m_tot], axis=1) + 1e-12
    return (bp / tot).astype(np.float32)  # [C]

def spectral_slope(psd_mean: np.ndarray, f: np.ndarray, fmin=2.0, fmax=45.0):
    """Approximate 1/f slope via linear fit in log-log."""
    m = (f>=fmin)&(f<=fmax)
    ff = f[m]; pp = psd_mean[m] + 1e-12
    x = np.log(ff); y = np.log(pp)
    # linear regression
    A = np.vstack([x, np.ones_like(x)]).T
    w = np.linalg.lstsq(A, y, rcond=None)[0]  # slope, intercept
    yhat = A @ w
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-12
    r2 = 1.0 - ss_res/ss_tot
    return float(w[0]), float(r2)  # slope, R^2

# ---------- Feature sets ----------
def feat_bp_only(x: np.ndarray, fs: int) -> np.ndarray:
    f, psd_c = psd_per_channel(x, fs)
    psd = psd_c.mean(0)
    feats=[]
    for lo,hi in BANDS_LEGACY.values():
        m = (f>=lo)&(f<hi)
        bp = np.trapz(psd[m], f[m])
        feats.append(bp)
    tot = np.trapz(psd[(f>=0.5)&(f<45.0)], f[(f>=0.5)&(f<45.0)]) + 1e-12
    feats = np.array(feats, np.float32) / tot
    return feats  # [4]

def feat_stat_only(x: np.ndarray) -> np.ndarray:
    mu = x.mean(1); sd = x.std(1) + 1e-12
    sk = ((x - mu[:,None])**3).mean(1) / (sd**3)
    ku = ((x - mu[:,None])**4).mean(1) / (sd**4)
    rms = np.sqrt((x**2).mean(1))
    ll  = np.abs(np.diff(x,1)).mean(1)
    return np.array([
        mu.mean(),mu.std(), sd.mean(),sd.std(),
        sk.mean(),sk.std(), ku.mean(),ku.std(),
        rms.mean(),rms.std(), ll.mean(),ll.std()
    ], np.float32)  # [12]

def feat_bp_stat(x: np.ndarray, fs: int) -> np.ndarray:
    return np.concatenate([feat_bp_only(x, fs), feat_stat_only(x)], 0)  # [16]

def feat_bp_stat_spatial(x: np.ndarray, fs: int, add_slope=False) -> np.ndarray:
    f, psd_c = psd_per_channel(x, fs)   # psd per channel
    feats=[]
    for lo,hi in BANDS_SPATIAL.values():
        bp_c = bandpower_chan(psd_c, f, lo, hi)  # [C]
        feats.extend([bp_c.mean(), bp_c.std()])  # mean&std across channels
    # add the legacy stats (time-domain) for shape
    feats.extend(feat_stat_only(x))
    if add_slope:
        slope, r2 = spectral_slope(psd_c.mean(0), f)
        feats.extend([slope, r2])
    return np.array(feats, np.float32)  # [ (len(BANDS_SPATIAL)*2) + 12 (+2) ] = 10+12(+2)=22(+2)

def featurize_batch(X: np.ndarray, fs: int, which: str, add_slope: bool, use_tqdm=False) -> np.ndarray:
    out=[]
    it = tqdmit(range(len(X)), use_tqdm, "featurize")
    for i in it:
        xi = X[i]
        if which == "bp_only":
            out.append(feat_bp_only(xi, fs))
        elif which == "stat_only":
            out.append(feat_stat_only(xi))
        elif which == "bp_stat":
            out.append(feat_bp_stat(xi, fs))
        else:  # bp_stat_spatial
            out.append(feat_bp_stat_spatial(xi, fs, add_slope))
    return np.stack(out, 0) if out else np.empty((0,1), np.float32)

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int) -> float:
    f1s=[]
    for c in range(n_cls):
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

def knn_predict(Ftr: np.ndarray, Ytr: np.ndarray, Fte: np.ndarray,
                k: int = 21, metric: str = "euclidean", weighted: bool = True) -> np.ndarray:
    if len(Ftr)==0 or len(Fte)==0: return np.empty((0,), np.int64)
    k = max(1, min(k, Ftr.shape[0]))
    if metric == "cosine":
        Ft = Ftr/(np.linalg.norm(Ftr, axis=1, keepdims=True)+1e-9)
        Fe = Fte/(np.linalg.norm(Fte, axis=1, keepdims=True)+1e-9)
        sims = Fe @ Ft.T
        idx = np.argpartition(-sims, min(k-1, sims.shape[1]-1), axis=1)[:, :k]
        rows = np.arange(Fe.shape[0])[:, None]
        w = sims[rows, idx] if weighted else np.ones_like(idx, np.float32)
    else:
        xx = (Fte**2).sum(1, keepdims=True)
        cc = (Ftr**2).sum(1)[None]
        xc = Fte @ Ftr.T
        d2 = xx + cc - 2*xc
        idx = np.argpartition(d2, min(k-1, d2.shape[1]-1), axis=1)[:, :k]
        rows = np.arange(Fte.shape[0])[:, None]
        w = 1.0/(np.sqrt(d2[rows, idx])+1e-9) if weighted else np.ones_like(idx, np.float32)
    neigh_y = Ytr[idx]
    n_cls = int(Ytr.max()) + 1
    pred = np.zeros((Fte.shape[0],), np.int64)
    for i in range(Fte.shape[0]):
        acc = np.bincount(neigh_y[i], weights=w[i], minlength=n_cls)
        pred[i] = int(np.argmax(acc))
    return pred

def main():
    args = parse_args()
    names = load_label_map(args.real_dir); n_cls=len(names); name2idx={n:i for i,n in enumerate(names)}
    print(f"[meta] classes: {names}")

    Xtr, Ytr = windows_from_dir(os.path.join(args.real_dir,"train"),
                                stride=max(1,args.stride), limit=0, use_tqdm=args.tqdm)
    if len(Xtr)==0: raise SystemExit("empty train split")
    keep = per_class_limits(Ytr, args.limit_train_per_class, n_cls)
    Xtr, Ytr = Xtr[keep], Ytr[keep]

    Xte, Yte = windows_from_dir(os.path.join(args.real_dir,"test"),
                                stride=max(1,args.stride), limit=args.limit_test, use_tqdm=args.tqdm)
    Xfk, _   = windows_from_dir(args.fake_dir, stride=max(1,args.stride),
                                limit=args.limit_fake, use_tqdm=args.tqdm)

    Ftr = featurize_batch(Xtr, fs=args.fs, which=args.feature, add_slope=args.add_spec_slope, use_tqdm=args.tqdm)
    Fte = featurize_batch(Xte, fs=args.fs, which=args.feature, add_slope=args.add_spec_slope, use_tqdm=args.tqdm) if len(Xte) else np.empty((0, Ftr.shape[1]), np.float32)
    Ffk = featurize_batch(Xfk, fs=args.fs, which=args.feature, add_slope=args.add_spec_slope, use_tqdm=args.tqdm) if len(Xfk) else np.empty((0, Ftr.shape[1]), np.float32)

    meta_std={}
    if args.standardize:
        mu = Ftr.mean(0, keepdims=True); sd = Ftr.std(0, keepdims=True) + 1e-9
        Ftr = (Ftr - mu)/sd
        if len(Fte): Fte = (Fte - mu)/sd
        if len(Ffk): Ffk = (Ffk - mu)/sd
        meta_std={"standardize": True}

    base={}
    if len(Fte):
        y_pred = knn_predict(Ftr, Ytr, Fte, k=args.k, metric=args.metric, weighted=args.weighted)
        base = {"acc": float((y_pred==Yte).mean()),
                "macro_f1": macro_f1(Yte, y_pred, n_cls),
                "n": int(len(Yte))}

    rec={}
    if len(Ffk):
        p_fake = knn_predict(Ftr, Ytr, Ffk, k=args.k, metric=args.metric, weighted=args.weighted)
        rec["n_fake"] = int(len(Ffk))
        if args.augment_artifact is not None:
            try:
                tgt = int(args.augment_artifact) if str(args.augment_artifact).isdigit() else name2idx[str(args.augment_artifact)]
                rec["intended_match"] = float((p_fake==tgt).mean())
            except Exception as e:
                rec["intended_match"] = None
                rec["_im_error"] = str(e)

    out = {"meta":{
                "class_names": names, "fs": args.fs,
                "feature": args.feature, "add_spec_slope": bool(args.add_spec_slope),
                "k": int(args.k), "metric": args.metric, "weighted": bool(args.weighted),
                "stride": int(args.stride),
                "limit_train_per_class": int(args.limit_train_per_class),
                "limit_test": int(args.limit_test),
                "limit_fake": int(args.limit_fake),
                **meta_std
            },
            "baseline": base, "recovery": rec}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
