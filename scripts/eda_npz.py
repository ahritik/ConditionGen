# scripts/eda_npz.py
import os, argparse, json, numpy as np
import matplotlib.pyplot as plt
from utils.constants import ARTIFACT_SET

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out/eda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.npz)
    art = d["artifact"]
    inten = d["intensity"].reshape(-1)
    X = d["x"]

    counts = {ARTIFACT_SET[i]: int((art==i).sum()) for i in range(len(ARTIFACT_SET))}
    with open(os.path.join(args.out_dir, "npz_summary.json"), "w") as f:
        json.dump({"counts": counts, "N": int(X.shape[0])}, f, indent=2)

    # class distribution plot
    plt.figure()
    names = list(counts.keys())
    vals = [counts[k] for k in names]
    plt.bar(names, vals)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "class_distribution.png"))
    print(f"Wrote EDA to {args.out_dir}")

if __name__ == "__main__":
    main()
