import csv, argparse, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_csv", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()
    steps, loss, mse, stft = [], [], [], []
    with open(args.log_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))
            mse.append(float(row["mse"]))
            stft.append(float(row.get("stft", "nan")) if row.get("stft","").strip() else float("nan"))
    plt.figure()
    plt.plot(steps, loss, label="loss")
    plt.plot(steps, mse, label="mse")
    if any([s==s for s in stft]):
        plt.plot(steps, stft, label="stft")
    plt.legend()
    plt.xlabel("step"); plt.ylabel("loss")
    plt.savefig(args.out_png, dpi=150)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
