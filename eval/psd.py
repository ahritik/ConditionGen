import os, glob, json, argparse, numpy as np
from scipy.signal import welch
from utils.constants import BANDS

def load_npz_dir(d, split="test"):
    files = sorted(glob.glob(os.path.join(d, f"{split}_*.npz")))
    Xs = []
    for f in files:
        with np.load(f) as z:
            Xs.append(z["x"])  # [N,C,T]
    if not Xs: return None
    X = np.concatenate(Xs, axis=0)
    return X

def load_fake_dir(d):
    # expects numpy array samples.npy
    x = np.load(os.path.join(d, "samples.npy"))
    return x

def band_power(x, fs, nperseg=256):
    # x: [N,C,T]
    N,C,T = x.shape
    freqs, Pxx = welch(x.reshape(-1,T), fs=fs, nperseg=nperseg, axis=-1)
    Pxx = Pxx.reshape(N,C,-1)
    bp = {}
    for name,(lo,hi) in BANDS.items():
        m = (freqs>=lo) & (freqs<hi)
        bp[name] = Pxx[..., m].mean(axis=-1)  # [N,C]
    return bp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="NPZ dir")
    ap.add_argument("--fake_dir", required=True, help="Samples dir with samples.npy")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out", type=str, default="out/metrics_psd.json")
    args = ap.parse_args()

    real = load_npz_dir(args.real_dir, split=args.split)
    fake = load_fake_dir(args.fake_dir)
    assert real is not None, "No real data found"
    # Match lengths
    n = min(len(real), len(fake))
    real = real[:n]; fake = fake[:n]

    r_bp = band_power(real, fs=args.fs)
    f_bp = band_power(fake, fs=args.fs)
    # Relative absolute error per band
    metrics = {}
    for b in BANDS.keys():
        r = r_bp[b].mean(axis=0) + 1e-8
        f = f_bp[b].mean(axis=0)
        rae = np.abs(f - r) / np.abs(r)
        metrics[b] = float(rae.mean())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
