# eval/intensity_trend.py
import os, re, json, glob, argparse, numpy as np
from scipy.stats import spearmanr, kendalltau
import torch, torch.nn as nn
from sklearn.metrics import auc

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

# --- tiny ResNet features/classifier (same as in classifier_eval) ---
class BasicBlock1D(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.proj  = nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False) if (c_in!=c_out or stride!=1) else None
    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        s = x if self.proj is None else self.proj(x)
        return torch.relu(y + s)

class ResNet1DTiny(nn.Module):
    def __init__(self, c_in=8, n_classes=7, widths=(32,64,128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, widths[0], 7, padding=3, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock1D(widths[0], widths[0], 1),
                                    BasicBlock1D(widths[0], widths[0], 1))
        self.layer2 = nn.Sequential(BasicBlock1D(widths[0], widths[1], 2),
                                    BasicBlock1D(widths[1], widths[1], 1))
        self.layer3 = nn.Sequential(BasicBlock1D(widths[1], widths[2], 2),
                                    BasicBlock1D(widths[2], widths[2], 1))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(widths[2], n_classes)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def _stack_npz(npz_dir, split):
    Xs, Ys = [], []
    for f in sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz"))):
        with np.load(f) as z:
            x = z["x"].astype(np.float32) if "x" in z.files else z[z.files[0]].astype(np.float32)
            y = None
            for k in ("y_artifact","a","y"):
                if k in z.files: y = z[k].astype(np.int64); break
            if y is None: raise RuntimeError(f"No labels in {f}")
            Xs.append(x); Ys.append(y)
    X = np.concatenate(Xs, axis=0); y = np.concatenate(Ys, axis=0)
    return X, y

def load_fake_dir(d, limit=None):
    npy = os.path.join(d,"samples.npy")
    if os.path.exists(npy):
        X = np.load(npy).astype(np.float32)
        return X if limit is None else X[:limit]
    Xs=[]
    for f in sorted(glob.glob(os.path.join(d,"*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else z.files[0]
            Xs.append(z[key].astype(np.float32))
    return np.concatenate(Xs,0) if Xs else np.zeros((0,8,1),np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--synth_parent", required=True, help="Folder containing subdirs like synth_eye_i40, synth_eye_i60, ...")
    ap.add_argument("--artifact", required=True, choices=ARTIFACT_SET)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_idx = ARTIFACT_SET.index(args.artifact)

    # Train a small classifier on real
    Xtr,ytr = _stack_npz(args.real_dir,"train")
    Xva,yva = _stack_npz(args.real_dir,"val")
    dev = pick_device()
    clf = ResNet1DTiny(c_in=Xtr.shape[1], n_classes=len(ARTIFACT_SET)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    def batches(X,y,bs,sh=True):
        idx = np.arange(len(X)); 
        if sh: np.random.shuffle(idx)
        for i in range(0,len(X),bs):
            j = idx[i:i+bs]
            yield torch.from_numpy(X[j]).to(dev), torch.from_numpy(y[j]).to(dev)

    clf.train()
    for ep in range(args.epochs):
        for xb,yb in batches(Xtr,ytr,args.batch,True):
            opt.zero_grad(); loss = crit(clf(xb), yb); loss.backward(); opt.step()
    clf.eval()

    # Discover intensity dirs
    pat = re.compile(rf"synth_{re.escape(args.artifact)}_i(\d+)$")
    subdirs = []
    for d in sorted(glob.glob(os.path.join(args.synth_parent, f"synth_{args.artifact}_i*"))):
        m = pat.search(os.path.basename(d))
        if m:
            inten = int(m.group(1))/100.0
            subdirs.append((inten, d))
    if not subdirs:
        raise SystemExit("No intensity subfolders like synth_eye_i40 found.")

    xs, ys = [], []
    for inten, d in subdirs:
        Xf = load_fake_dir(d)
        with torch.no_grad():
            probs=[]
            for i in range(0,len(Xf),args.batch):
                xb = torch.from_numpy(Xf[i:i+args.batch]).to(dev)
                pr = clf(xb).softmax(dim=1).cpu().numpy()
                probs.append(pr[:,target_idx])
            p = float(np.concatenate(probs).mean()) if probs else 0.0
        xs.append(inten); ys.append(p)

    # monotonicity stats
    order = np.argsort(xs)
    xs = np.array(xs)[order]; ys = np.array(ys)[order]
    rho, _ = spearmanr(xs, ys)
    tau, _ = kendalltau(xs, ys)
    area = float(auc(xs, ys))  # higher is better

    out = {
        "artifact": args.artifact,
        "points": [{"intensity": float(x), "p_intended": float(y)} for x,y in zip(xs,ys)],
        "spearman_rho": float(rho if np.isfinite(rho) else 0.0),
        "kendall_tau": float(tau if np.isfinite(tau) else 0.0),
        "auc_intensity_prob": area
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
