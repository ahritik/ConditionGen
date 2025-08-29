import os, glob, json, argparse, numpy as np

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
    x = np.load(os.path.join(d, "samples.npy"))
    return x

def chan_cov(X):
    # covariance per sample; average over samples
    covs = []
    for i in range(X.shape[0]):
        xi = X[i]  # [C,T]
        covs.append(np.cov(xi))
    return np.stack(covs).mean(axis=0)

def acf(x, nlags=100):
    x = x - x.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1: len(x)+nlags-1]
    ac /= (np.arange(len(ac))[::-1] + 1e-6)
    ac /= ac[0] + 1e-8
    return ac

def mean_acf(X, nlags=100):
    acs = []
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            acs.append(acf(X[i,c], nlags=nlags))
    return np.stack(acs).mean(axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--out", type=str, default="out/metrics_cov_acf.json")
    args = ap.parse_args()

    real = load_npz_dir(args.real_dir, split=args.split)
    fake = load_fake_dir(args.fake_dir)
    assert real is not None, "No real data found"
    n = min(len(real), len(fake))
    real = real[:n]; fake = fake[:n]

    rcov = chan_cov(real)
    fcov = chan_cov(fake)
    cov_dist = float(np.linalg.norm(rcov - fcov, ord='fro')) / real.shape[1]

    racf = mean_acf(real)
    facf = mean_acf(fake)
    acf_dist = float(np.linalg.norm(racf - facf) / len(racf))

    metrics = {"cov_fro": cov_dist, "acf_l2": acf_dist}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
