# scripts/plot_training.py
import os, argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ax = df.plot(x="step", y=["loss","loss_mse","loss_stft"])
    plt.title("Training Losses")
    plt.tight_layout()
    if args.out_png:
        plt.savefig(args.out_png)
        print(f"Wrote {args.out_png}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
