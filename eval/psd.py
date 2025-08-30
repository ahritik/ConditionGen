# eval/psd.py
import os, glob, json, argparse, numpy as np
from scipy.signal import welch
from utils.constants import BANDS  # e.g., {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}

def _stack_npz_dir(npz_dir: str, split: str, limit: int | None = None) -> np.ndarray:
    Xs, total = [], 0
    for f in sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz"))):
        with np.load(f) as z:
            x = z["x"]  # [N,C,T]
            Xs.append(x)
            total += x.shape[0]
            if limit is not None and total >= limit:
                break
    if not Xs:
        return np.zeros((0, 8, 1), np.float32)
    X = np.concatenate(Xs, axis=0)
    if limit is not None and X.shape[0] > limit:
        X = X[:limit]
    return X.astype(np.float32)

def load_fake_dir(d: str, limit: int | None = None) -> np.ndarray:
    """Load generated windows from either samples.npy or NPZ shards with key 'x'."""
    npy = os.path.join(d, "samples.npy")
    if os.path.exists(npy):
        X = np.load(npy).astype(np.float32)
        if limit is not None and X.shape[0] > limit:
            X = X[:limit]
        return X
    # fall back to NPZ shards
    Xs = []
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
            if key is None:
                continue
            x = z[key].astype(np.float32)
            Xs.append(x)
            n += x.shape[0]
            if limit is not None and n >= limit:
                break
    return np.concatenate(Xs, axis=0) if Xs else np.zeros((0,8,1), np.float32)

def band_powers(X: np.ndarray, fs: int, nperseg: int | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute mean Welch PSD on channel-mean, then average within canonical bands."""
    if X.shape[0] == 0:
        return {k: float("nan") for k in BANDS}, np.array([]), np.array([])
    # channel-mean per window
    Xm = X.mean(axis=1)  # [N,T]
    T = Xm.shape[-1]
    if nperseg is None:
        nperseg = min(512, T)
    freqs, Pxx = welch(Xm, fs=fs, nperseg=nperseg)  # Pxx: [N,F]
    psd_mean = Pxx.mean(axis=0)                     # [F]
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        bp[name] = float(psd_mean[m].mean()) if np.any(m) else float("nan")
    return bp, freqs, psd_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--fs", type=int, default=None, help="override fs; else read meta.json")
    ap.add_argument("--limit_real", type=int, default=None)
    ap.add_argument("--limit_fake", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # fs from meta if not given
    if args.fs is None:
        try:
            meta = json.load(open(os.path.join(args.real_dir, "meta.json")))
            fs = int(meta.get("fs", 200))
        except Exception:
            fs = 200
    else:
        fs = args.fs

    real = _stack_npz_dir(args.real_dir, args.split, limit=args.limit_real)
    fake = load_fake_dir(args.fake_dir, limit=args.limit_fake)

    bp_real, fr, psd_r = band_powers(real, fs)
    bp_fake, ff, psd_f = band_powers(fake, fs)

    band_rel_err = {k: float(abs(bp_fake[k] - bp_real[k]) / (abs(bp_real[k]) + 1e-8))
                    for k in BANDS.keys()}

    out = {
        "fs": fs,
        "n_real": int(real.shape[0]),
        "n_fake": int(fake.shape[0]),
        "band_power_real": bp_real,
        "band_power_fake": bp_fake,
        "band_rel_err": band_rel_err
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
