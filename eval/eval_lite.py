#!/usr/bin/env python3
# eval/eval_lite.py
import os, glob, json, argparse, numpy as np
from collections import defaultdict
from scipy.signal import welch

def load_real(real_dir, max_per_art=3000):
    # search recursively for .npz
    paths = sorted(glob.glob(os.path.join(real_dir, "**", "*.npz"), recursive=True))
    buckets = defaultdict(list)  # artifact_id or "all" -> list of (N,C,T)
    any_loaded = False

    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as z:
                key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
                if key is None:
                    continue
                X = z[key].astype(np.float32)
                if X.ndim == 2:  # (C,T)
                    X = X[None, ...]
                any_loaded = True

                y = z.get("y_artifact", None)
                if y is None:
                    buckets["all"].append(X)
                else:
                    y = np.array(y).reshape(-1)
                    n = min(len(X), len(y))
                    if n <= 0:
                        continue
                    X = X[:n]
                    y = y[:n]
                    for aid in np.unique(y):
                        sel = X[y == aid]
                        if sel.size:
                            buckets[int(aid)].append(sel)
        except Exception:
            # skip unreadable files
            continue

    if not any_loaded or not buckets:
        raise RuntimeError(f"No usable real samples found under {real_dir}. "
                           f"Check the path and that .npz contain arrays under keys like 'x'.")

    # stack and cap per bucket
    real = {}
    rng = np.random.default_rng(0)
    for k, v in buckets.items():
        X = np.concatenate(v, 0)
        if X.shape[0] > max_per_art:
            idx = rng.choice(X.shape[0], max_per_art, replace=False)
            X = X[idx]
        real[k] = X
    return real

def bandpowers(x, fs=200.0, nperseg=512):
    # x: (C,T) -> 4-bandpower mean across channels
    f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    def bp(lo,hi): 
        m=(f>=lo)&(f<hi); 
        return P[...,m].mean(axis=-1).mean()
    return np.array([bp(0.5,4), bp(4,8), bp(8,13), bp(13,30)], dtype=np.float32)

def acf_vec(x, max_lag=50):
    # x: (C,T) -> mean ACF across channels, first max_lag lags (excluding lag0 scaling)
    C,T = x.shape
    x = x - x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True) + 1e-8
    acfs=[]
    for c in range(C):
        r = np.correlate(x[c], x[c], mode="full")[T-1:T-1+max_lag] / (v[c,0]*np.arange(T, T-max_lag, -1))
        acfs.append(r)
    return np.mean(np.stack(acfs,0),0).astype(np.float32)  # (L,)

def cov_mat(x):
    # x: (C,T) -> (C,C) covariance across time
    x = x - x.mean(axis=-1, keepdims=True)
    return (x @ x.T) / (x.shape[-1]-1)

def feature_vec(x, fs=200.0):
    # compact per-sample feature for distributional tests
    bp = bandpowers(x, fs=fs)             # (4,)
    ac = acf_vec(x, max_lag=32)           # (32,)
    chstd = x.std(axis=-1).mean(0)[()]    # scalar: mean per-channel std
    return np.concatenate([bp, ac, [chstd]], 0)

def fro_norm(A,B): 
    D = A-B; return float(np.sqrt((D*D).sum()))

def l2(v,w): 
    d=v-w; return float(np.sqrt((d*d).sum()))

def one_nn_two_sample(Fr, Ff):
    # simple half/half split 1-NN acc
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    n=min(len(Fr), len(Ff), 2000)
    rng=np.random.default_rng(0)
    ir=rng.choice(len(Fr), n, replace=False)
    iff=rng.choice(len(Ff), n, replace=False)
    X=np.vstack([Fr[ir], Ff[iff]])
    y=np.array([0]*n + [1]*n)
    idx=rng.permutation(2*n); tr=idx[:n]; te=idx[n:]
    knn=KNeighborsClassifier(n_neighbors=1)
    knn.fit(X[tr], y[tr])
    yhat=knn.predict(X[te])
    return float(accuracy_score(y[te], yhat))

def prd_knn(Fr, Ff, k=5):
    from sklearn.neighbors import NearestNeighbors
    nbr = NearestNeighbors(n_neighbors=k).fit(Fr)
    d_r, _ = nbr.kneighbors(Fr)
    rad = d_r[:, -1]
    d_f, idx = nbr.kneighbors(Ff, n_neighbors=1)
    precision = float((d_f[:,0] <= rad[idx[:,0]]).mean())
    nbr_f = NearestNeighbors(n_neighbors=1).fit(Ff)
    d_rf, _ = nbr_f.kneighbors(Fr)
    recall = float((d_rf[:,0] <= rad).mean())
    return precision, recall

def summarize(real_X, fake_X, fs=200.0):
    # real_X, fake_X: (N,C,T)
    # --- PSD bands
    Rb = np.array([bandpowers(x, fs) for x in real_X])
    Fb = np.array([bandpowers(x, fs) for x in fake_X])
    psd_delta = np.abs(Rb.mean(0) - Fb.mean(0)).tolist()
    # --- ACF
    Ra = np.array([acf_vec(x) for x in real_X])
    Fa = np.array([acf_vec(x) for x in fake_X])
    acf_l2 = l2(Ra.mean(0), Fa.mean(0))
    # --- Covariance
    Rc = np.array([cov_mat(x) for x in real_X])
    Fc = np.array([cov_mat(x) for x in fake_X])
    cov_fro = fro_norm(Rc.mean(0), Fc.mean(0))
    # --- distributional (features)
    Rf = np.array([feature_vec(x, fs) for x in real_X])
    Ff = np.array([feature_vec(x, fs) for x in fake_X])
    one_nn = one_nn_two_sample(Rf, Ff)
    prec, rec = prd_knn(Rf, Ff)
    return {
        "psd_delta": {"delta": psd_delta[0], "theta": psd_delta[1], "alpha": psd_delta[2], "beta": psd_delta[3]},
        "acf_l2": acf_l2,
        "cov_fro": cov_fro,
        "one_nn_acc": one_nn,
        "knn_precision": prec,
        "knn_recall": rec,
        "n_real": int(real_X.shape[0]),
        "n_fake": int(fake_X.shape[0])
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_parent", required=True)  # folder with synth_* subdirs
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    real = load_real(args.real_dir)
    # choose reference real pool: if artifact labels exist use 0..K-1, else "all"
    artifact_keys = [k for k in real.keys() if k!="all"]
    have_labels = len(artifact_keys)>0
    if not have_labels:
        print("No y_artifact in real; using global real pool for all.")
        Xr = real.get("all", None)
        if Xr is None:
            # fall back to concatenating whatever buckets we have
            Xr = np.concatenate([real[k] for k in real], 0)

    report={}
    lines=[]
    lines.append("# Eval-Lite Summary\n")
    lines.append("| Artifact | Δδ | Δθ | Δα | Δβ | Cov Fro ↓ | ACF L2 ↓ | 1-NN acc ↓ | PRD-Precision ↑ | PRD-Recall ↑ | n_fake |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for d in sorted(glob.glob(os.path.join(args.fake_parent, "synth_*"))):
        art = os.path.basename(d).split("synth_")[-1]
        p = os.path.join(d, "samples.npy")
        if not os.path.exists(p): 
            continue
        Xf = np.load(p).astype(np.float32)
        if Xf.ndim==2: Xf = Xf[None,...]
        # pick matching real set
        if have_labels:
            # try to map artifact name to an integer id by name order if present
            # default: use all real if we can't find a bucket
            try:
                # heuristic: common 6-class mapping
                name2id = {"none":0,"eye":1,"muscle":2,"chewing":3,"shiver":4,"electrode":5}
                aid = name2id.get(art, None)
                Xr = real.get(aid, None)
            except Exception:
                Xr = None
        else:
            Xr = real["all"]
        if Xr is None:
            # fallback to global
            Xr = np.concatenate([real[k] for k in real], 0)

        # subsample to balance compute
        n = min(len(Xr), 2000)
        if len(Xr)>n:
            idx = np.random.default_rng(0).choice(len(Xr), n, replace=False)
            Xr = Xr[idx]
        nf = min(len(Xf), 2000)
        if len(Xf)>nf:
            idx = np.random.default_rng(1).choice(len(Xf), nf, replace=False)
            Xf = Xf[idx]

        res = summarize(Xr, Xf, fs=args.fs)
        report[art]=res
        lines.append(f"| {art} | {res['psd_delta']['delta']:.3f} | {res['psd_delta']['theta']:.3f} | {res['psd_delta']['alpha']:.3f} | {res['psd_delta']['beta']:.3f} | {res['cov_fro']:.3f} | {res['acf_l2']:.3f} | {res['one_nn_acc']:.3f} | {res['knn_precision']:.3f} | {res['knn_recall']:.3f} | {res['n_fake']} |")

    md = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        with open(os.path.splitext(args.out)[0]+".json","w") as f:
            json.dump(report,f,indent=2)
    print(md)

if __name__ == "__main__":
    main()
