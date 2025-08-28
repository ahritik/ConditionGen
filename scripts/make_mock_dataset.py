# scripts/make_mock_dataset.py
import os, argparse, numpy as np
from utils.constants import ARTIFACT_SET, MONTAGE_IDS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--T", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    rng = np.random.default_rng(0)

    x = rng.normal(size=(args.N, args.C, args.T)).astype(np.float32)
    # Make a few pseudo-artifacts by adding low-freq drifts (eye) or bursts (muscle)
    artifact = rng.integers(0, len(ARTIFACT_SET), size=(args.N,), dtype=np.int64)
    intensity = rng.random((args.N,1)).astype(np.float32)
    seizure = rng.integers(0,2,size=(args.N,1)).astype(np.float32)
    age_bin = rng.integers(0,4,size=(args.N,), dtype=np.int64)
    montage_id = np.full((args.N,), MONTAGE_IDS["canon8"], dtype=np.int64)

    np.savez_compressed(args.out_npz,
        x=x,
        artifact=artifact,
        intensity=intensity,
        seizure=seizure,
        age_bin=age_bin,
        montage_id=montage_id
    )
    print(f"Wrote {args.out_npz}")

if __name__ == "__main__":
    main()
