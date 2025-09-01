#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier evaluation on TUAR-style NPZ shards.
- Recovery mode: train on real train/val, report baseline on real test,
  then predict on fakes in --fake_dir and report intended_match, counts, etc.
- Augmentation mode: same baseline; then augment TRAIN with fakes from --fake_dir
  *labeled as --augment_artifact* and retrain; report Δmetrics.
Includes tqdm progress bars with auto/forced control.

Examples
--------
# Recovery (predict intended artifact on fakes)
python -m eval.classifier_eval \
  --real_dir out/npz \
  --fake_dir out/eval_run_XXXX/synth_eye \
  --task artifact --arch resnet1d \
  --epochs 8 --batch 256 --lr 1e-3 \
  --out out/clf_eval/recovery_eye_resnet1d.json

# Augmentation (train+synthetic, re-train)
python -m eval.classifier_eval \
  --real_dir out/npz \
  --fake_dir out/eval_run_XXXX/synth_electrode \
  --augment_artifact electrode \
  --task artifact --arch eegnet \
  --epochs 8 --batch 256 --lr 1e-3 \
  --out out/clf_eval/augment_gain_electrode_eegnet.json
"""
# eval/classifier_eval.py
import argparse, json, os, glob, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

# -------------------------------
# Taxonomy: keep 7-class head (trained shape),
# but we only *evaluate/augment* with these 6:
EVAL_ARTS = ["none","eye","muscle","chewing","shiver","electrode"]
CLASS_NAMES = EVAL_ARTS + ["movement"]  # movement kept for head shape only
NAME2IDX = {n:i for i,n in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)  # 7
# -------------------------------

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def device_pick():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def load_npz_split(root, split, limit=None, label_key="y_artifact"):
    """Load TUAR shards: expects keys x, y_artifact (int64)."""
    root = Path(root)
    files = sorted(root.glob(f"{split}_*.npz"))
    xs, ys = [], []
    for fp in files:
        d = np.load(fp)
        x = d["x"]  # (N,C,T)
        # label resolution
        if label_key in d:
            y = d[label_key].astype(np.int64)
        elif "artifact" in d:
            y = d["artifact"].astype(np.int64)
        elif "y" in d:
            y = d["y"].astype(np.int64)
        else:
            raise KeyError(f"No labels found in {fp} (looked for {label_key}, artifact, y)")
        xs.append(x); ys.append(y)
        if limit is not None and sum(t.shape[0] for t in xs) >= limit:
            break
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if limit is not None and X.shape[0] > limit:
        X = X[:limit]; y = y[:limit]
    return X, y

def load_fake_dir(fake_dir):
    """Load generated samples from synth folder (prefers samples.npy)."""
    cand = ["samples.npy", "samples_post.npy", "x.npy"]
    for c in cand:
        p = Path(fake_dir) / c
        if p.exists():
            x = np.load(p, mmap_mode=None)
            return x
    raise FileNotFoundError(f"No fake array found in {fake_dir} (looked for {cand})")

# ----- simple models
class Tiny1D(nn.Module):
    def __init__(self, in_ch=8, ncls=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, ncls)
    def forward(self, x):
        # x: (B,C,T)
        h = self.net(x).squeeze(-1)
        return self.fc(h)

class ResNet1D(nn.Module):
    def __init__(self, in_ch=8, ncls=NUM_CLASSES, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3), nn.BatchNorm1d(width), nn.ReLU()
        )
        self.block1 = self._blk(width, width)
        self.block2 = self._blk(width, width*2, stride=2)
        self.block3 = self._blk(width*2, width*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(width*2, ncls)
    def _blk(self, c1, c2, stride=1):
        return nn.Sequential(
            nn.Conv1d(c1, c2, 3, stride=stride, padding=1), nn.BatchNorm1d(c2), nn.ReLU(),
            nn.Conv1d(c2, c2, 3, padding=1), nn.BatchNorm1d(c2), nn.ReLU(),
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class EEGNetLite(nn.Module):
    """Compact EEGNet-like conv stack (not a strict reimplementation)."""
    def __init__(self, in_ch=8, ncls=NUM_CLASSES):
        super().__init__()
        self.conv_t = nn.Conv2d(1, 8, (1, 32), padding=(0, 16))
        self.conv_s = nn.Conv2d(8, 16, (in_ch, 1), groups=8)
        self.dw = nn.Conv2d(16, 32, (1, 16), padding=(0,8), groups=16)
        self.pw = nn.Conv2d(32, 64, 1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, ncls)
    def forward(self, x):
        # x: (B,C,T) -> (B,1,C,T)
        x = x.unsqueeze(1)
        x = F.elu(self.conv_t(x))
        x = F.elu(self.conv_s(x))
        x = F.elu(self.dw(x))
        x = F.elu(self.pw(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

def make_model(arch, in_ch=8, ncls=NUM_CLASSES):
    if arch == "tiny":
        return Tiny1D(in_ch, ncls)
    if arch == "resnet1d":
        return ResNet1D(in_ch, ncls)
    if arch == "eegnet":
        return EEGNetLite(in_ch, ncls)
    raise ValueError(arch)

def class_weights(y_np):
    # robust: y_np is (N,)
    counts = np.bincount(y_np, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts==0] = 1.0
    w = 1.0 / counts
    w = w * (NUM_CLASSES / w.sum())
    return torch.tensor(w, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    bs = 512
    preds = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs]).to(device)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    yhat = np.concatenate(preds, axis=0)
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y, yhat, labels=list(range(NUM_CLASSES)))
    # per-class f1 dictionary (by name)
    per_f1 = {}
    for k, name in enumerate(CLASS_NAMES):
        f = f1_score((y==k).astype(int), (yhat==k).astype(int), zero_division=0)
        per_f1[name] = float(f)
    return {"macro_f1": float(f1), "acc": float(acc), "confusion": cm.tolist(), "per_class_f1": per_f1, "n_test": int(y.shape[0])}

def train_model(model, Xtr, ytr, Xva, yva, device, epochs=10, lr=1e-3, class_weight="balanced", pbar=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if class_weight == "balanced":
        w = class_weights(ytr).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    # numpy -> torch datasets
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds   = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)

    best = {"acc": -1, "state": None}
    loop = range(epochs)
    if pbar is not None:
        loop = pbar(range(epochs), desc="train", leave=False)
    for _ in loop:
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        # quick val
        with torch.no_grad():
            model.eval()
            all_logits, all_y = [], []
            for xb, yb in va_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu()); all_y.append(yb)
            logits = torch.cat(all_logits,0); yv = torch.cat(all_y,0).numpy()
            yh = logits.argmax(1).numpy()
            acc = accuracy_score(yv, yh)
            if acc > best["acc"]:
                best["acc"] = acc
                best["state"] = {k: v.cpu() for k,v in model.state_dict().items()}
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="out/npz")
    ap.add_argument("--fake_dir", help=".../synth_X")
    ap.add_argument("--augment_artifact", choices=EVAL_ARTS, help="Artifact name for augmentation & recovery IM")
    ap.add_argument("--augment_with", choices=EVAL_ARTS, help="Alias to augment_artifact")
    ap.add_argument("--task", choices=["artifact"], default="artifact")
    ap.add_argument("--arch", choices=["tiny","resnet1d","eegnet"], default="resnet1d")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)  # kept for compatibility
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--class_weight", choices=["none","balanced"], default="balanced")
    ap.add_argument("--label_key", default="y_artifact")
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_pick()

    # resolve target name
    target_name = args.augment_with if args.augment_with is not None else args.augment_artifact
    if (args.fake_dir is None) != (target_name is None):
        # either both None (baseline-only) or both set (recovery/augmentation)
        pass

    # load splits
    Xtr, ytr = load_npz_split(args.real_dir, "train", args.limit_train, label_key=args.label_key)
    Xva, yva = load_npz_split(args.real_dir, "val",   args.limit_val,   label_key=args.label_key)
    Xte, yte = load_npz_split(args.real_dir, "test",  args.limit_test,  label_key=args.label_key)

    # standardize dtype and shape for torch
    Xtr = Xtr.astype(np.float32); Xva = Xva.astype(np.float32); Xte = Xte.astype(np.float32)

    pbar = tqdm if args.tqdm else None
    model = make_model(args.arch, in_ch=Xtr.shape[1], ncls=NUM_CLASSES)
    best = train_model(model, Xtr, ytr, Xva, yva, device, args.epochs, args.lr, args.class_weight, pbar)
    model.load_state_dict(best["state"])

    # baseline test metrics
    baseline = evaluate(model, Xte, yte, device)

    out = {"arch": args.arch, "baseline": baseline}

    # recovery / augmentation (if fake_dir is provided)
    if args.fake_dir and target_name:
        target_idx = NAME2IDX[target_name]  # maps into 7-way head
        Xfake = load_fake_dir(args.fake_dir).astype(np.float32)
        with torch.no_grad():
            model.eval()
            bs = 512; preds = []
            loop = range(0, Xfake.shape[0], bs)
            if pbar: loop = pbar(loop, desc=f"recovery:{target_name}", leave=False)
            for i in loop:
                xb = torch.from_numpy(Xfake[i:i+bs]).to(device)
                logits = model(xb)
                preds.append(logits.argmax(1).cpu().numpy())
            yhat = np.concatenate(preds, 0)

        # Intended match = fraction predicted as target_idx
        im = float((yhat == target_idx).mean()) if yhat.size else 0.0
        # counts by class name
        counts = {name: int((yhat == idx).sum()) for name, idx in NAME2IDX.items()}

        out["recovery"] = {
            "target": target_name,
            "target_idx": target_idx,
            "n_fake": int(Xfake.shape[0]),
            "intended_match": im,
            "pred_counts": counts,
        }

        # augmentation: add fake -> train
        yfake = np.full((Xfake.shape[0],), target_idx, dtype=np.int64)
        Xtr_aug = np.concatenate([Xtr, Xfake], 0)
        ytr_aug = np.concatenate([ytr, yfake], 0)

        model_aug = make_model(args.arch, in_ch=Xtr.shape[1], ncls=NUM_CLASSES)
        best_aug = train_model(model_aug, Xtr_aug, ytr_aug, Xva, yva, device, args.epochs, args.lr, args.class_weight, pbar)
        model_aug.load_state_dict(best_aug["state"])
        aug = evaluate(model_aug, Xte, yte, device)

        out["augmentation"] = {
            "augment_artifact": target_name,
            "n_train_base": int(Xtr.shape[0]),
            "n_train_aug": int(Xtr_aug.shape[0]),
            "macro_f1_base": float(baseline["macro_f1"]),
            "macro_f1_aug": float(aug["macro_f1"]),
            "delta_macro_f1": float(aug["macro_f1"] - baseline["macro_f1"]),
            "acc_base": float(baseline["acc"]),
            "acc_aug": float(aug["acc"]),
            "delta_acc": float(aug["acc"] - baseline["acc"]),
        }

    # write
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}")
    if "recovery" in out:
        print(f"IM={out['recovery']['intended_match']:.3f}, n_fake={out['recovery']['n_fake']}")
    print(f"Baseline macro-F1={baseline['macro_f1']:.3f} acc={baseline['acc']:.3f}")

if __name__ == "__main__":
    main()
