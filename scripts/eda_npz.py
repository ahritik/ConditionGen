import os, glob, json, argparse, numpy as np
from utils.constants import ARTIFACT_SET

def summarize(npz_dir):
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    total = 0
    art_counts = {k:0 for k in ARTIFACT_SET}
    intens = []
    for f in files:
        with np.load(f) as z:
            n = z["x"].shape[0]; total += n
            a = z["y_artifact"]
            for idx in a:
                art_counts[ARTIFACT_SET[int(idx)]] += 1
            intens.append(z["intensity"])
    intens = np.concatenate(intens) if len(intens)>0 else np.zeros(0)
    return {"n_windows": int(total), "artifact_counts": art_counts, "intensity_hist": np.histogram(intens, bins=10)[0].tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    s = summarize(args.npz_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "npz_summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))

if __name__=="__main__":
    main()
