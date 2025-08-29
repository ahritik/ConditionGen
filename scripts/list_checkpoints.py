import os, argparse, re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    args = ap.parse_args()
    for f in sorted(os.listdir(args.ckpt_dir)):
        if f.endswith(".pt"):
            m = re.search(r"step_(\d+)(_ema)?\.pt", f)
            if m:
                step = int(m.group(1)); ema = bool(m.group(2))
                print(f"{f}\tstep={step}\tema={ema}")

if __name__ == "__main__":
    main()
