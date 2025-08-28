# scripts/eda_tuar.py
import os, argparse, json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuar_root", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="out/eda/tuar_summary.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    total_csv = 0
    total_rows = 0
    labels = {}

    for dirpath, dirnames, filenames in os.walk(args.tuar_root):
        for f in filenames:
            if f.endswith(".csv"):
                total_csv += 1
                p = os.path.join(dirpath, f)
                try:
                    df = pd.read_csv(p)
                except:
                    continue
                total_rows += len(df)
                if "label" in df.columns:
                    for v, cnt in df["label"].value_counts().to_dict().items():
                        labels[v] = labels.get(v, 0) + int(cnt)

    summary = {"csv_files": total_csv, "rows": total_rows, "label_counts": labels}
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
