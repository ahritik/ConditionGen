# eval/clf_artifact.py
"""
Train a tiny CNN artifact classifier and report condition recovery accuracy.
"""
import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.constants import ARTIFACT_SET

class TinyCNN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, len(ARTIFACT_SET))

    def forward(self, x):
        h = self.net(x)  # (B,64,1)
        h = h.squeeze(-1)
        return self.fc(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_npz", type=str, required=True)
    ap.add_argument("--synth_npz", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    r = np.load(args.real_npz)
    s = np.load(args.synth_npz)

    xr = torch.tensor(r["x"], dtype=torch.float32)
    yr = torch.tensor(r["artifact"], dtype=torch.long)
    xs = torch.tensor(s["x"], dtype=torch.float32)
    ys = torch.tensor(s.get("artifact", np.zeros(xs.shape[0], dtype=np.int64)), dtype=torch.long)

    C = xr.shape[1]
    model = TinyCNN(C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    # train on real
    dl = DataLoader(TensorDataset(xr, yr), batch_size=args.batch, shuffle=True)
    model.train()
    for ep in range(args.epochs):
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            opt.step()

    # evaluate on synth
    model.eval()
    with torch.no_grad():
        logits = model(xs)
        pred = logits.argmax(dim=1)
        acc = (pred == ys).float().mean().item()
    print(f"Condition recovery accuracy on synth: {acc:.4f}")

if __name__ == "__main__":
    main()
