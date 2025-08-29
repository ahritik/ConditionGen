import os, glob, json, argparse, pandas as pd
from utils.constants import ARTIFACT_SET

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    edfs = sorted(glob.glob(os.path.join(args.tuar_root, "**/*.edf"), recursive=True))
    csvs = [os.path.splitext(e)[0]+".csv" for e in edfs]
    n_csv = sum(os.path.exists(c) for c in csvs)
    stats = {"n_edf": len(edfs), "n_csv": int(n_csv)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tuar_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(stats)

if __name__ == "__main__":
    main()
