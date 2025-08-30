# eval/psd_extra.py
import os, json, glob, argparse, numpy as np
from scipy.signal import welch
from scipy.stats import wasserstein_distance

BANDS = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}

def load_real(npz_dir, split, limit=None):
    Xs = []
    for f in sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz"))):
        with np.load(f) as z:
            x = z["x"].astype(np.float32) if "x" in z.files else z[z.files[0]].astype(np.float32)
            Xs.append(x)
            if limit and sum(xx.shape[0] for xx in Xs) >= limit: break
    return np.concatenate(Xs,0) if Xs else np.zeros((0,8,1),np.float32)

def load_fake(d, limit=None):
    p = os.path.join(d,"samples.npy")
    if os.path.exists(p):
        X = np.load(p).astype(np.float32)
        return X if limit is None else X[:limit]
    Xs=[]
    for f in sorted(glob.glob(os.path.join(d,"*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else z.files[0]
            Xs.append(z[key].astype(np.float32))
    X = np.concatenate(Xs,0) if Xs else np.zeros((0,8,1),np.float32)
    return X if limit is None else X[:limit]

def mean_psd(X, fs=200):
    if len(X)==0: return None, None
    N,C,T = X.shape
    nperseg = min(512, T)
    XC = X.reshape(N*C, T)
    f, P = welch(XC, fs=fs, nperseg=nperseg)  # [N*C,F]
    return f, P.mean(0)

def cosine(a,b):
    num = float((a*b).sum())
    den = float(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
    return num/den

def band_rel_err(freqs, psd_r, psd_f):
    out={}
    for k,(lo,hi) in BANDS.items():
        m=(freqs>=lo)&(freqs<hi)
        r=psd_r[m].mean(); f=psd_f[m].mean()
        out[k]=float(abs(f-r)/(abs(r)+1e-12))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Xr = load_real(args.real_dir, args.split, limit=args.limit)
    Xf = load_fake(args.fake_dir, args.limit)

    n = min(len(Xr), len(Xf))
    Xr = Xr[:n]; Xf = Xf[:n]

    fr, Pr = mean_psd(Xr, args.fs)
    ff, Pf = mean_psd(Xf, args.fs)
    if fr is None or ff is None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"error":"empty"}, open(args.out,"w"), indent=2)
        print(f"Wrote {args.out} (empty)")
        return

    # cosine similarity (↑ better)
    cos = float(cosine(Pr, Pf))
    # 1D EMD over frequency axis (↓ better)
    # normalize spectra to sum=1 to behave like distributions
    r = Pr.clip(1e-12); r = r/r.sum()
    f = Pf.clip(1e-12); f = f/f.sum()
    emd = float(wasserstein_distance(fr, ff, u_weights=r, v_weights=f))
    # band-wise relative errors (for context)
    bre = band_rel_err(fr, Pr, Pf)

    out = {"cosine_psd": cos, "emd_psd": emd, "band_rel_err": bre,
           "n": int(n), "fs": args.fs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=2)
    print(f"Wrote {args.out}")
if __name__ == "__main__":
    main()
