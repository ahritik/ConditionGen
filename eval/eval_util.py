#!/usr/bin/env python3
import argparse, os, json, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from tqdm import tqdm
from collections import Counter

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

class NpzDataset(Dataset):
    def __init__(self, npz_path, split_seed=13, split=(0.7,0.15,0.15), part="train"):
        z = np.load(npz_path, allow_pickle=True)
        X = z["X"]; y=z["y"]; subj=z["subject"]
        # split by subject
        subs = sorted(set(subj.tolist()))
        rng = np.random.RandomState(split_seed); rng.shuffle(subs)
        n=len(subs); ntr=int(n*split[0]); nv=int(n*split[1])
        tr=set(subs[:ntr]); va=set(subs[ntr:ntr+nv]); te=set(subs[ntr+nv:])
        if part=="train":
            keep=[i for i in range(len(X)) if subj[i] in tr]
        elif part=="val":
            keep=[i for i in range(len(X)) if subj[i] in va]
        else:
            keep=[i for i in range(len(X)) if subj[i] in te]
        self.X=X[keep].astype(np.float32); self.y=y[keep].astype(np.int64)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TinyCNN(nn.Module):
    def __init__(self, C, ncls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, ncls)
    def forward(self, x):
        h = self.net(x).squeeze(-1)
        return self.fc(h)

def train_eval(train_ds, test_ds, lr=1e-3, epochs=5, device=None):
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    C = train_ds[0][0].shape[0]; ncls=len(ARTIFACT_SET)
    net = TinyCNN(C, ncls).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    tr_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    te_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    for ep in range(epochs):
        net.train()
        for xb,yb in tr_dl:
            xb=torch.tensor(xb,device=dev); yb=torch.tensor(yb,device=dev)
            opt.zero_grad(); loss=crit(net(xb), yb); loss.backward(); opt.step()
    # eval
    net.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in te_dl:
            xb=torch.tensor(xb,device=dev)
            ys.extend(yb.tolist())
            ps.extend(net(xb).argmax(-1).cpu().tolist())
    macro = f1_score(ys, ps, average="macro")
    return macro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=str, required=True)
    ap.add_argument("--synthetic", type=str, required=True, help="folder of npz samples")
    ap.add_argument("--augment_rare", action="store_true")
    ap.add_argument("--report", type=str, default="out/utility.json")
    args = ap.parse_args()

    train_real = NpzDataset(args.real, part="train")
    test_real  = NpzDataset(args.real, part="test")

    # Baseline
    f1_base = train_eval(train_real, test_real, epochs=5)

    # Build augmented dataset
    if args.augment_rare:
        # count real class distribution
        y = train_real.y
        cnt = Counter(y.tolist())
        maj = max(cnt.values())
        # load synthetic
        Xs=[]; ys=[]
        for fn in os.listdir(args.synthetic):
            if fn.endswith(".npz"):
                z = np.load(os.path.join(args.synthetic, fn), allow_pickle=True)
                X = z["X"].astype(np.float32)
                art = z.get("artifact")
                if art is None: art = fn.split("_")[0]
                yidx = ARTIFACT_SET.index(str(art).lower())
                Xs.append(X); ys.extend([yidx]*len(X))
        if len(Xs)==0:
            raise SystemExit("No synthetic files found")
        Xsyn = np.concatenate(Xs,0); ysyn=np.array(ys,np.int64)
        # oversample rare classes up to maj
        addX=[]; addy=[]
        for cls in range(len(ARTIFACT_SET)):
            need = max(0, maj - cnt.get(cls,0))
            if need==0: continue
            idx = np.where(ysyn==cls)[0]
            if len(idx)==0: continue
            sel = np.random.choice(idx, size=min(need, len(idx)), replace=len(idx)<need)
            addX.append(Xsyn[sel]); addy.extend([cls]*len(sel))
        if addX:
            addX = np.concatenate(addX,0); addy=np.array(addy,np.int64)
            train_real.X = np.concatenate([train_real.X, addX],0)
            train_real.y = np.concatenate([train_real.y, addy],0)

    f1_aug = train_eval(train_real, test_real, epochs=5)

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report,"w") as f:
        json.dump({"macro_f1_baseline": f1_base, "macro_f1_augmented": f1_aug, "delta_macro_f1": f1_aug-f1_base}, f, indent=2)
    print("Utility Δmacro-F1:", f1_aug - f1_base)

if __name__ == "__main__":
    main()
