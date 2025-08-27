#!/usr/bin/env python3
import argparse, os, json, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from tqdm import tqdm

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

class NpzWindows(Dataset):
    def __init__(self, path, train_real=False, eval_generated=False):
        if train_real:
            z = np.load(path, allow_pickle=True)
            self.X = z["X"].astype(np.float32); self.y = z["y"].astype(np.int64)
        elif eval_generated:
            # folder with multiple .npz, each labeled via metadata (artifact field) or filename
            Xs=[]; ys=[]
            for fn in os.listdir(path):
                if fn.endswith(".npz"):
                    z = np.load(os.path.join(path,fn), allow_pickle=True)
                    X = z["X"].astype(np.float32)
                    art = z.get("artifact")
                    if art is None:
                        # infer from filename
                        art = fn.split("_")[0]
                    y = ARTIFACT_SET.index(str(art).lower())
                    Xs.append(X); ys.extend([y]*len(X))
            self.X = np.concatenate(Xs,0); self.y=np.array(ys,np.int64)
        else:
            z = np.load(path, allow_pickle=True)
            self.X = z["X"].astype(np.float32); self.y = z["y"].astype(np.int64)
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

def train_classifier(train_path, report_path=None, device=None):
    z = np.load(train_path, allow_pickle=True)
    X=z["X"]; y=z["y"]
    C=X.shape[1]; ncls=len(ARTIFACT_SET)
    ds = NpzWindows(train_path, train_real=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    net = TinyCNN(C, ncls).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(3):
        net.train()
        for xb,yb in dl:
            xb = torch.tensor(xb, device=dev); yb=torch.tensor(yb, device=dev)
            opt.zero_grad()
            loss = crit(net(xb), yb); loss.backward(); opt.step()
    # save
    os.makedirs("ckpts", exist_ok=True)
    torch.save(net.state_dict(), "ckpts/artifact_clf.pt")
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path,"w") as f: f.write(json.dumps({"status":"trained"}, indent=2))
    print("Classifier trained and saved to ckpts/artifact_clf.pt")

def eval_generated(gen_folder, report_path=None, device=None, real_path=None):
    ds = NpzWindows(gen_folder, eval_generated=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    # infer C from real if provided
    if real_path is not None:
        z = np.load(real_path, allow_pickle=True); C=z["X"].shape[1]
    else:
        # assume 8
        C=8
    net = TinyCNN(C, len(ARTIFACT_SET)).to(dev)
    net.load_state_dict(torch.load("ckpts/artifact_clf.pt", map_location=dev))
    net.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for xb, yb in dl:
            xb = torch.tensor(xb, device=dev)
            logits = net(xb)
            ys.extend(yb.tolist())
            ps.extend(logits.argmax(dim=-1).cpu().tolist())
    f1 = f1_score(ys, ps, average="macro")
    report = classification_report(ys, ps, target_names=ARTIFACT_SET, output_dict=True)
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path,"w") as f: json.dump({"macro_f1": f1, "report": report}, f, indent=2)
    print("Generated specificity macro-F1:", f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="NPZ (for train) or folder (for gen)")
    ap.add_argument("--train_real", action="store_true")
    ap.add_argument("--eval_generated", action="store_true")
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--real_path", type=str, default=None)
    args = ap.parse_args()
    if args.train_real:
        train_classifier(args.data, report_path=args.report)
    elif args.eval_generated:
        eval_generated(args.data, report_path=args.report, real_path=args.real_path)
    else:
        print("Specify --train_real or --eval_generated")

if __name__ == "__main__":
    main()
