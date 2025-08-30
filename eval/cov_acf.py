# eval/cov_acf.py
import os, glob, json, argparse, numpy as np

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
    npy = os.path.join(d, "samples.npy")
    if os.path.exists(npy):
        X = np.load(npy).astype(np.float32)
        if limit is not None and X.shape[0] > limit:
            X = X[:limit]
        return X
    Xs = []
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
            if key is None: continue
            x = z[key].astype(np.float32)
            Xs.append(x)
            n += x.shape[0]
            if limit is not None and n >= limit:
                break
    return np.concatenate(Xs, axis=0) if Xs else np.zeros((0,8,1), np.float32)

def mean_covariance(X: np.ndarray) -> np.ndarray:
    """Mean channel covariance over windows."""
    covs = [np.cov(X[i]) for i in range(X.shape[0])]
    return np.stack(covs).mean(axis=0) if covs else np.zeros((X.shape[1], X.shape[1]), np.float32)

def acf(x: np.ndarray, nlags=150) -> np.ndarray:
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x)-1 : len(x)+nlags-1]
    ac /= (np.arange(len(ac))[::-1] + 1e-6)
    ac /= ac[0] + 1e-8
    return ac

def mean_acf_first_channel(X: np.ndarray, nlags=150) -> np.ndarray:
    if X.shape[0] == 0: return np.zeros((nlags,), np.float32)
    acs = [acf(X[i,0], nlags=nlags) for i in range(X.shape[0])]
    return np.stack(acs).mean(axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--limit_real", type=int, default=None)
    ap.add_argument("--limit_fake", type=int, default=None)
    ap.add_argument("--nlags", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    real = _stack_npz_dir(args.real_dir, args.split, limit=args.limit_real)
    fake = load_fake_dir(args.fake_dir, limit=args.limit_fake)

    C_r = mean_covariance(real)
    C_f = mean_covariance(fake)
    cov_fro = float(np.linalg.norm(C_f - C_r, ord="fro"))

    A_r = mean_acf_first_channel(real, nlags=args.nlags)
    A_f = mean_acf_first_channel(fake, nlags=args.nlags)
    acf_l2 = float(np.linalg.norm(A_f - A_r))

    out = {"cov_fro": cov_fro, "acf_l2": acf_l2,
           "n_real": int(real.shape[0]), "n_fake": int(fake.shape[0])}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
