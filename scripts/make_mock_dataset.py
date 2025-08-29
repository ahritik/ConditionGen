import os, argparse, numpy as np
from utils.constants import ARTIFACT_SET

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--fs", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    N = args.n
    C = 8
    T = 800  # 4s@200
    rng = np.random.default_rng(0)
    X = rng.normal(0,1,(N,C,T)).astype(np.float32)
    # simple band-limited patterns per artifact
    art = rng.integers(0, len(ARTIFACT_SET), size=N, endpoint=False)
    seiz = rng.integers(0,2,size=N)
    ageb = rng.integers(0,4,size=N)
    mont = np.zeros(N, dtype=np.int64)
    inten = rng.random(N).astype(np.float32)
    np.savez(os.path.join(args.out_dir, "train_000.npz"),
             x=X[:int(0.6*N)], y_artifact=art[:int(0.6*N)], y_seizure=seiz[:int(0.6*N)],
             y_agebin=ageb[:int(0.6*N)], y_montage=mont[:int(0.6*N)], intensity=inten[:int(0.6*N)])
    np.savez(os.path.join(args.out_dir, "val_000.npz"),
             x=X[int(0.6*N):int(0.8*N)], y_artifact=art[int(0.6*N):int(0.8*N)], y_seizure=seiz[int(0.6*N):int(0.8*N)],
             y_agebin=ageb[int(0.6*N):int(0.8*N)], y_montage=mont[int(0.6*N):int(0.8*N)], intensity=inten[int(0.6*N):int(0.8*N)])
    np.savez(os.path.join(args.out_dir, "test_000.npz"),
             x=X[int(0.8*N):], y_artifact=art[int(0.8*N):], y_seizure=seiz[int(0.8*N):],
             y_agebin=ageb[int(0.8*N):], y_montage=mont[int(0.8*N):], intensity=inten[int(0.8*N):])
    print("Mock NPZ dataset created at", args.out_dir)

if __name__ == "__main__":
    main()
