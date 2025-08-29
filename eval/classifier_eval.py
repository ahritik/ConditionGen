import os, glob, json, argparse, numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.constants import ARTIFACT_SET

class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class TinyCNN(nn.Module):
    def __init__(self, c=8, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_cls)
        )
    def forward(self, x): return self.net(x)

def load_real(npz_dir, task="artifact", split="train"):
    files = sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz")))
    Xs, y = [], []
    for f in files:
        with np.load(f) as z:
            Xs.append(z["x"])
            if task=="artifact":
                y.append(z["y_artifact"])
            elif task=="seizure":
                y.append(z["y_seizure"])
            else:
                raise ValueError("task must be artifact or seizure")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def load_fake(fake_dir, n=None, label=None, task="artifact"):
    X = np.load(os.path.join(fake_dir, "samples.npy"))
    if n is not None: X = X[:n]
    if label is None:
        # default: none artifact or non-seizure
        label = 0 if task=="artifact" else 0
    y = np.full(len(X), label, dtype=np.int64)
    return X, y

def train_eval(Xtr, ytr, Xte, yte, n_cls):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = TinyCNN(n_cls=n_cls).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr_loader = DataLoader(NumpyDataset(Xtr,ytr), batch_size=64, shuffle=True)
    te_loader = DataLoader(NumpyDataset(Xte,yte), batch_size=64, shuffle=False)
    for ep in range(3):
        m.train()
        for xb,yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = m(xb)
            loss = nn.CrossEntropyLoss()(logits, yb)
            loss.backward()
            opt.step()
    m.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb,yb in te_loader:
            xb = xb.to(dev)
            logits = m(xb)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            ps.append(proba)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_proba = np.concatenate(ps)
    y_pred = y_proba.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    if n_cls == 2:
        auroc = roc_auc_score(y_true, y_proba[:,1])
    else:
        # one-vs-rest macro AUROC
        aurocs = []
        for k in range(n_cls):
            yb = (y_true==k).astype(np.int32)
            aurocs.append(roc_auc_score(yb, y_proba[:,k]))
        auroc = float(np.mean(aurocs))
    return macro_f1, auroc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", default="")
    ap.add_argument("--augment_with", default="")
    ap.add_argument("--task", choices=["artifact","seizure"], default="artifact")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_cls = len(ARTIFACT_SET) if args.task=="artifact" else 2
    Xtr, ytr = load_real(args.real_dir, task=args.task, split="train")
    Xte, yte = load_real(args.real_dir, task=args.task, split="test")

    res = {}
    # baseline
    f1, auc = train_eval(Xtr, ytr, Xte, yte, n_cls)
    res["baseline"] = {"macroF1": float(f1), "AUROC": float(auc)}

    # specificity: train classifier and evaluate on FAKE labeled by intended condition (if provided)
    if args.fake_dir:
        Xf, yf = load_fake(args.fake_dir, label=None, task=args.task)
        # Assume all fake share the same label encoded in fake meta (optional)
        meta_p = os.path.join(args.fake_dir, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p) as f: meta = json.load(f)
            if args.task=="artifact":
                lbl = ARTIFACT_SET.index(meta.get("artifact","none"))
            else:
                lbl = int(meta.get("seizure", 0))
            yf[:] = lbl
        f1s, aucs = train_eval(Xtr, ytr, Xf, yf, n_cls)
        res["specificity_recovery"] = {"macroF1": float(f1s), "AUROC": float(aucs)}

    # utility: augment training with FAKE then test on REAL
    if args.augment_with:
        Xf, yf = load_fake(args.augment_with, label=None, task=args.task)
        Xtr_aug = np.concatenate([Xtr, Xf], axis=0)
        ytr_aug = np.concatenate([ytr, yf], axis=0)
        f1a, auca = train_eval(Xtr_aug, ytr_aug, Xte, yte, n_cls)
        res["utility_augmented"] = {
            "macroF1": float(f1a), "AUROC": float(auca),
            "delta_macroF1": float(f1a - res["baseline"]["macroF1"]),
            "delta_AUROC": float(auca - res["baseline"]["AUROC"]),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
