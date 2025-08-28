# scripts/list_checkpoints.py
import os, re, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="out")
    args = ap.parse_args()
    for f in sorted(os.listdir(args.dir)):
        if f.startswith("ckpt_") and f.endswith(".pt"):
            m = re.findall(r"ckpt_(\d+)\.pt", f)
            step = m[0] if m else "?"
            print(f"{f}\tstep={step}")
if __name__ == "__main__":
    main()
