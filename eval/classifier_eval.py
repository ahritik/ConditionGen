#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifier evaluation on TUAR-style NPZ shards (6 classes, no 'movement').

Modes
-----
1) Baseline: train on real train/val; report metrics on real test.
2) Recovery: baseline + predict on fakes in --fake_dir; report intended_match and counts.
3) Augmentation: baseline + re-train with fakes labeled as --augment_artifact; report deltas.

Examples
--------
# Recovery (predict intended artifact on fakes)
python -m eval.classifier_eval \
  --real_dir out/npz \
  --fake_dir out/eval_run_XXXX/synth_eye \
  --augment_artifact eye \
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

import argparse, json, os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm

# ---------------------------------------------------------------------
# Final 6-class taxonomy (movement REMOVED completely)
ARTIFACT_SET = ["none", "eye", "muscle", "chewing", "shiver", "electrode"]
NAME2IDX = {n: i for i, n in enumerate(ARTIFACT_SET)}
IDX2NAME = {i: n for n, i in NAME2IDX.items()}
NUM_CLASSES = len(ARTIFACT_SET)  # 6
# ---------------------------------------------------------------------


# -------------------- utils --------------------
def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def device_pick():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def _concat_until_limit(xs, ys, limit):
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if limit is not None and X.shape[0] > limit:
        X = X[:limit]; y = y[:limit]
    return X, y


def _filter_to_6_classes(X, y):
    """Keep only labels in {0..5}. Drop anything else (e.g., old 'movement'=6)."""
    keep = (y >= 0) & (y < NUM_CLASSES)
    if keep.sum() != y.shape[0]:
        dropped = int(y.shape[0] - keep.sum())
        print(f"[filter] dropping {dropped} samples outside 6-class set")
    return X[keep], y[keep]


def load_npz_split(root, split, limit=None, label_key="y_artifact"):
    """
    Load TUAR shards: keys: x, y_artifact (preferred) or artifact or y.
    Returns X:(N,C,T) float32; y:(N,) int64. Filters to 6 classes.
    """
    root = Path(root)
    files = sorted(root.glob(f"{split}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No shards found at {root}/{split}_*.npz")
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
    X, y = _concat_until_limit(xs, ys, limit)
    X, y = _filter_to_6_classes(X, y)
    X = X.astype(np.float32)
    return X, y


def load_fake_dir(fake_dir):
    """Load generated samples from synth folder (prefers samples.npy)."""
    cand = ["samples.npy", "samples_post.npy", "x.npy"]
    for c in cand:
        p = Path(fake_dir) / c
        if p.exists():
            x = np.load(p, mmap_mode=None).astype(np.float32)
            if x.ndim != 3:
                raise ValueError(f"Fake array must be (N,C,T), got shape {x.shape} in {p}")
            return x
    raise FileNotFoundError(f"No fake array found in {fake_dir} (looked for {cand})")


def hist(y):
    h = {IDX2NAME[i]: int((y == i).sum()) for i in range(NUM_CLASSES)}
    h["total"] = int(y.shape[0])
    return h


# -------------------- simple models --------------------
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
    """Compact EEGNet-ish conv stack (shape-stable for (B,C,T))."""
    def __init__(self, in_ch=8, ncls=NUM_CLASSES):
        super().__init__()
        self.conv_t = nn.Conv2d(1, 8, (1, 32), padding=(0, 16))
        self.conv_s = nn.Conv2d(8, 16, (in_ch, 1), groups=8)
        self.dw = nn.Conv2d(16, 32, (1, 16), padding=(0, 8), groups=16)
        self.pw = nn.Conv2d(32, 64, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, ncls)
    def forward(self, x):
        x = x.unsqueeze(1)             # (B,1,C,T)
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
    counts = np.bincount(y_np, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w * (NUM_CLASSES / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, X, y, device, bs=512):
    model.eval()
    preds = []
    for i in range(0, X.shape[0], bs):
        xb = torch.from_numpy(X[i:i+bs]).to(device)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    yhat = np.concatenate(preds, axis=0)
    acc = accuracy_score(y, yhat)
    cm = confusion_matrix(y, yhat, labels=list(range(NUM_CLASSES)))
    f1_macro = f1_score(y, yhat, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)

    per_f1 = {}
    for k in range(NUM_CLASSES):
        per_f1[IDX2NAME[k]] = float(
            f1_score((y == k).astype(int), (yhat == k).astype(int), zero_division=0)
        )

    return {
        "macro_f1": float(f1_macro),
        "acc": float(acc),
        "confusion": cm.tolist(),
        "per_class_f1": per_f1,
        "n_test": int(y.shape[0]),
    }


def train_model(model, Xtr, ytr, Xva, yva, device, epochs=8, lr=1e-3, class_weight="balanced", pbar=None, bs_train=256, bs_val=512):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if class_weight == "balanced":
        w = class_weights(ytr).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds   = torch.utils.data.TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs_train, shuffle=True, drop_last=False)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs_val, shuffle=False, drop_last=False)

    best = {"acc": -1.0, "state": None}
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
            logits = torch.cat(all_logits, 0); yv = torch.cat(all_y, 0).numpy()
            yh = logits.argmax(1).numpy()
            acc = accuracy_score(yv, yh)
            if acc > best["acc"]:
                best["acc"] = acc
                best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="Directory with TUAR shards (train_*.npz, val_*.npz, test_*.npz)")
    ap.add_argument("--fake_dir", help=".../synth_{artifact} (must contain samples.npy or x.npy)")
    ap.add_argument("--augment_artifact", choices=ARTIFACT_SET, help="Target artifact name for recovery + augmentation")
    ap.add_argument("--task", choices=["artifact"], default="artifact")
    ap.add_argument("--arch", choices=["tiny", "resnet1d", "eegnet"], default="resnet1d")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="balanced")
    ap.add_argument("--label_key", default="y_artifact")
    ap.add_argument("--tqdm", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = device_pick()

    # Mode validation
    do_recovery_or_aug = (args.fake_dir is not None) and (args.augment_artifact is not None)

    # Load data (and filter to 6 classes)
    Xtr, ytr = load_npz_split(args.real_dir, "train", args.limit_train, label_key=args.label_key)
    Xva, yva = load_npz_split(args.real_dir, "val",   args.limit_val,   label_key=args.label_key)
    Xte, yte = load_npz_split(args.real_dir, "test",  args.limit_test,  label_key=args.label_key)

    # Report histograms (helpful sanity check)
    print("Class histogram (after filtering to 6 classes):")
    print("  train:", hist(ytr))
    print("  val  :", hist(yva))
    print("  test :", hist(yte))

    # Build & train baseline
    pbar = tqdm if args.tqdm else None
    in_ch = Xtr.shape[1]
    model = make_model(args.arch, in_ch=in_ch, ncls=NUM_CLASSES)
    best = train_model(model, Xtr, ytr, Xva, yva, device,
                       epochs=args.epochs, lr=args.lr,
                       class_weight=args.class_weight,
                       pbar=pbar, bs_train=args.batch, bs_val=max(512, args.batch))
    model.load_state_dict(best["state"])

    # Baseline test
    baseline = evaluate(model, Xte, yte, device, bs=max(512, args.batch))
    out = {"arch": args.arch, "baseline": baseline}

    # Recovery + Augmentation
    if do_recovery_or_aug:
        target = args.augment_artifact
        target_idx = NAME2IDX[target]

        # Recovery: predict fakes
        Xfake = load_fake_dir(args.fake_dir)
        with torch.no_grad():
            model.eval()
            preds = []
            loop = range(0, Xfake.shape[0], max(512, args.batch))
            if pbar: loop = pbar(loop, desc=f"recovery:{target}", leave=False)
            for i in loop:
                xb = torch.from_numpy(Xfake[i:i+max(512, args.batch)]).to(device)
                logits = model(xb)
                preds.append(logits.argmax(1).cpu().numpy())
            yhat = np.concatenate(preds, 0)

        im = float((yhat == target_idx).mean()) if yhat.size else 0.0
        counts = {name: int((yhat == idx).sum()) for name, idx in NAME2IDX.items()}

        out["recovery"] = {
            "target": target,
            "target_idx": target_idx,
            "n_fake": int(Xfake.shape[0]),
            "intended_match": im,
            "pred_counts": counts,
        }

        # Augmentation: add fakes with target label to train
        yfake = np.full((Xfake.shape[0],), target_idx, dtype=np.int64)
        Xtr_aug = np.concatenate([Xtr, Xfake], 0)
        ytr_aug = np.concatenate([ytr, yfake], 0)

        model_aug = make_model(args.arch, in_ch=in_ch, ncls=NUM_CLASSES)
        best_aug = train_model(model_aug, Xtr_aug, ytr_aug, Xva, yva, device,
                               epochs=args.epochs, lr=args.lr,
                               class_weight=args.class_weight,
                               pbar=pbar, bs_train=args.batch, bs_val=max(512, args.batch))
        model_aug.load_state_dict(best_aug["state"])
        aug = evaluate(model_aug, Xte, yte, device, bs=max(512, args.batch))

        out["augmentation"] = {
            "augment_artifact": target,
            "n_train_base": int(Xtr.shape[0]),
            "n_train_aug": int(Xtr_aug.shape[0]),
            "macro_f1_base": float(baseline["macro_f1"]),
            "macro_f1_aug": float(aug["macro_f1"]),
            "delta_macro_f1": float(aug["macro_f1"] - baseline["macro_f1"]),
            "acc_base": float(baseline["acc"]),
            "acc_aug": float(aug["acc"]),
            "delta_acc": float(aug["acc"] - baseline["acc"]),
        }

    # Write results
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}")
    if "recovery" in out:
        print(f"IM={out['recovery']['intended_match']:.3f}, n_fake={out['recovery']['n_fake']}")
    print(f"Baseline macro-F1={baseline['macro_f1']:.3f} acc={baseline['acc']:.3f}")


if __name__ == "__main__":
    main()
