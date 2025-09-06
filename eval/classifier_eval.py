#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/classifier_eval.py
-----------------------
Train a compact 1D classifier on real TUAR NPZ windows and evaluate:
  • baseline: metrics on real test
  • recovery: metrics on fake windows (+ intended_match if --augment_artifact)

Features
--------
• tqdm progress bars (toggle with --tqdm).
• Spawn-safe on macOS/MPS (no nested functions; workers=0 unless CUDA).
• Fast-mode controls:
    - --stride_train / --stride_eval : take every k-th window
    - --limit_train/val/test/fake    : cap number of items
    - big batch / few epochs for quick passes
• Device override: --device {auto,cpu,cuda,mps}
• Class names auto-loaded from label_map.json if present; fallback to 6 TUAR classes.

Usage examples
--------------
# Quick all-in-one (tiny, fast):
python -m eval.classifier_eval \
  --real_dir out/tuar_npz \
  --fake_dir out/eval_run/synth_eye \
  --augment_artifact eye \
  --arch tiny --epochs 3 --batch 1024 --lr 1e-3 \
  --stride_train 4 --stride_eval 4 --limit_train 60000 --limit_test 10000 --limit_fake 8000 \
  --out out/clf_eval/recovery_eye_tiny.json --tqdm
"""

from __future__ import annotations
import os, json, glob, argparse
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------ Defaults / utils ------------------------------

FALLBACK_CLASSES = ["none", "eye", "muscle", "chewing", "shiver", "electrode"]

def load_label_map(*roots: str) -> List[str]:
    """
    Try to load class names from label_map.json in any of the given roots.
    Falls back to 6-class TUAR order if not found.
    """
    for r in roots:
        if not r: 
            continue
        cands = []
        if os.path.isdir(r):
            cands.append(os.path.join(r, "label_map.json"))
        else:
            cands.append(os.path.join(os.path.dirname(r), "label_map.json"))
        for c in cands:
            if os.path.exists(c):
                try:
                    j = json.load(open(c))
                    arr = j.get("artifact_names")
                    if isinstance(arr, list) and arr:
                        return arr
                except Exception:
                    pass
    return FALLBACK_CLASSES

def pick_device(forced: str = "auto") -> torch.device:
    if forced != "auto":
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def workers_and_pin(device: torch.device) -> Tuple[int, bool]:
    # Multiprocessing + pinned memory helpful mainly on CUDA
    if device.type == "cuda":
        return 4, True
    return 0, False

def iter_with_tqdm(it: Iterable, enabled: bool, desc: str):
    """Wrap iterable with tqdm if enabled; otherwise return as-is."""
    if not enabled:
        return it
    try:
        from tqdm import tqdm  # lazy import
        return tqdm(it, desc=desc, dynamic_ncols=True, leave=False)
    except Exception:
        return it

# ------------------------------ Dataset ---------------------------------------

class DirNPZDataset(Dataset):
    """
    Stream all *.npz under a folder and return (x, y) for classification.
    Fast-mode options:
      - stride: use every k-th window within each npz
      - limit_items: cap total items after indexing
    """
    def __init__(self, folder: str, label_key: str = "y_artifact",
                 stride: int = 1, limit_items: int = 0):
        super().__init__()
        self.folder = folder
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No NPZ shards in {folder}")
        self.label_key = label_key
        self.stride = max(1, int(stride))
        self.index: List[Tuple[int, int]] = []  # (file_idx, row_idx)
        self._cache: Optional[Tuple[int, np.lib.npyio.NpzFile]] = None

        # Build index with stride; stop early if limit reached
        remaining = int(limit_items) if limit_items else None
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                n = int(z["x"].shape[0])
            for i in range(0, n, self.stride):
                self.index.append((fi, i))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining is not None and remaining <= 0:
                break

    def __len__(self) -> int:
        return len(self.index)

    def _open(self, fi: int):
        return np.load(self.files[fi], allow_pickle=True)

    def __getitem__(self, idx: int):
        fi, li = self.index[idx]
        if self._cache is None or self._cache[0] != fi:
            if self._cache is not None:
                try:
                    self._cache[1].close()
                except Exception:
                    pass
            self._cache = (fi, self._open(fi))
        z = self._cache[1]
        x = z["x"][li].astype(np.float32)  # [C,T]
        # label
        if self.label_key in z:
            y = z[self.label_key][li]
        else:
            if "artifact" in z: y = z["artifact"][li]
            elif "y_artifact" in z: y = z["y_artifact"][li]
            else: raise KeyError("No artifact/y_artifact label in shard")
        return torch.from_numpy(x), int(y)

# ------------------------------ Models ----------------------------------------

class TinyConv1D(nn.Module):
    """Lightweight baseline 1D CNN."""
    def __init__(self, c_in: int, n_cls: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_cls)
    def forward(self, x):  # [B,C,T]
        h = self.net(x).squeeze(-1)  # [B,128]
        return self.head(h)

class ResBlock1D(nn.Module):
    def __init__(self, c: int, k: int = 5, d: int = 1):
        super().__init__()
        pad = ((k - 1) * d) // 2
        self.conv1 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.act1  = nn.ReLU()
        self.conv2 = nn.Conv1d(c, c, k, padding=pad, dilation=d)
        self.act2  = nn.ReLU()
    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        return self.act2(h + x)

class ResNet1D(nn.Module):
    def __init__(self, c_in: int, n_cls: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, 64, 7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block1 = ResBlock1D(64, 5, 1)
        self.block2 = ResBlock1D(64, 5, 2)
        self.block3 = ResBlock1D(64, 5, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, n_cls)
    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h); h = self.block2(h); h = self.block3(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h)

class EEGNetLite(nn.Module):
    """Very small EEGNet-style model (depthwise + pointwise)."""
    def __init__(self, c_in: int, n_cls: int):
        super().__init__()
        self.conv_time = nn.Conv1d(c_in, 16, kernel_size=9, padding=4, groups=c_in)  # depthwise
        self.pw1 = nn.Conv1d(16, 64, kernel_size=1)                                  # pointwise
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(2)
        self.dw2 = nn.Conv1d(64, 64, kernel_size=9, padding=4, groups=64)
        self.pw2 = nn.Conv1d(64, 64, kernel_size=1)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, n_cls)
    def forward(self, x):
        h = self.act(self.pw1(self.conv_time(x)))
        h = self.pool(h)
        h = self.act(self.pw2(self.dw2(h)))
        h = self.avg(h).squeeze(-1)
        return self.head(h)

def make_model(arch: str, c_in: int, n_cls: int) -> nn.Module:
    arch = (arch or "tiny").lower()
    if arch == "tiny":     return TinyConv1D(c_in, n_cls)
    if arch == "resnet1d": return ResNet1D(c_in, n_cls)
    if arch == "eegnet":   return EEGNetLite(c_in, n_cls)
    raise ValueError(f"Unknown arch: {arch}")

# ------------------------------ Metrics ---------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             use_tqdm: bool = False, desc: str = "eval") -> Dict[str, float]:
    """Compute accuracy and macro-F1 over a loader with an optional tqdm bar."""
    model.eval()
    all_pred, all_true = [], []
    iterator = iter_with_tqdm(loader, use_tqdm, desc)
    for x, y in iterator:
        x = x.to(device); y = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        all_pred.append(pred.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
    if not all_pred:
        return {"acc": 0.0, "macro_f1": 0.0, "n": 0}
    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    n_cls = int(y_pred.max()) + 1 if y_pred.size else 0
    acc = float((y_pred == y_true).mean()) if y_true.size else 0.0
    # macro-F1
    f1s = []
    for c in range(n_cls):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(float(f1))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {"acc": acc, "macro_f1": macro_f1, "n": int(y_true.size)}

@torch.no_grad()
def intended_match(model: nn.Module, loader: DataLoader, device: torch.device,
                   target_idx: int, use_tqdm: bool = False) -> Tuple[float,int]:
    """Fraction of predictions equal to target_idx over all items in loader."""
    model.eval()
    total = 0
    match = 0
    iterator = iter_with_tqdm(loader, use_tqdm, f"intended_match={target_idx}")
    for x, _ in iterator:
        x = x.to(device)
        pred = model(x).argmax(dim=1).detach().cpu().numpy()
        match += int((pred == target_idx).sum())
        total += int(pred.size)
    return (match / max(1, total), total)

# ------------------------------ Train loop ------------------------------------

def train_model(model: nn.Module, loader: DataLoader, device: torch.device,
                epochs: int, lr: float, use_tqdm: bool = False) -> None:
    """Simple supervised training loop with optional tqdm."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        it = iter_with_tqdm(loader, use_tqdm, desc=f"epoch {ep}/{epochs}")
        model.train()
        for x, y in it:
            x = x.to(device)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

# ------------------------------ Argparse & main --------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    # Real data: either pass a single base with splits, or explicit paths
    ap.add_argument("--real_dir", default=None, help="base dir containing train/val/test subfolders")
    ap.add_argument("--real_train", default=None)
    ap.add_argument("--real_val",   default=None)
    ap.add_argument("--real_test",  default=None)
    ap.add_argument("--fake_dir",   required=True)

    ap.add_argument("--label_key", default="y_artifact", help="label key to read from shards")
    ap.add_argument("--artifact_names", default=None, help="optional path to label_map.json or comma list")
    ap.add_argument("--augment_artifact", default=None,
                    help="artifact name or index expected in fake_dir (for intended_match)")

    # Fast-mode subsampling
    ap.add_argument("--limit_train", type=int, default=0, help="max train items (0=all)")
    ap.add_argument("--limit_val",   type=int, default=0, help="max val items (0=all)")
    ap.add_argument("--limit_test",  type=int, default=0, help="max test items (0=all)")
    ap.add_argument("--limit_fake",  type=int, default=0, help="max fake items (0=all)")
    ap.add_argument("--stride_train", type=int, default=1, help="use every k-th train window")
    ap.add_argument("--stride_eval",  type=int, default=1, help="use every k-th window for val/test/fake")

    # System / training
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="force device if desired")
    ap.add_argument("--arch", default="tiny", choices=["tiny","resnet1d","eegnet"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch",  type=int, default=256)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--tqdm", action="store_true", help="show progress bars")

    ap.add_argument("--out", required=True, help="output JSON")
    return ap.parse_args()

def resolve_real_dirs(args) -> Tuple[str,str,str]:
    if args.real_dir:
        base = args.real_dir
        rt = os.path.join(base, "train")
        rv = os.path.join(base, "val")
        rtest = os.path.join(base, "test")
        return rt, rv, rtest
    return args.real_train, args.real_val, args.real_test

def main():
    args = parse_args()
    device = pick_device(args.device)
    n_workers, pin_mem = workers_and_pin(device)

    # Resolve dirs
    real_train, real_val, real_test = resolve_real_dirs(args)
    if not (real_train and os.path.isdir(real_train)):
        raise SystemExit(f"real_train missing/invalid: {real_train}")
    if not (real_val and os.path.isdir(real_val)):
        print("[warn] real_val missing — proceeding without validation.")
        real_val = None
    if not (real_test and os.path.isdir(real_test)):
        print("[warn] real_test missing — proceeding without baseline test.")
        real_test = None
    if not os.path.isdir(args.fake_dir):
        raise SystemExit(f"fake_dir missing/invalid: {args.fake_dir}")

    # Class names
    if args.artifact_names:
        if os.path.isfile(args.artifact_names):
            names = json.load(open(args.artifact_names)).get("artifact_names", FALLBACK_CLASSES)
        else:
            names = [s.strip() for s in args.artifact_names.split(",") if s.strip()]
    else:
        names = load_label_map(real_train, real_val, real_test, args.fake_dir)
    n_cls = len(names)
    name2idx = {n:i for i,n in enumerate(names)}
    print(f"[meta] class order: {names}")

    # Datasets & loaders (spawn-safe on macOS/MPS; fast on CUDA)
    train_ds = DirNPZDataset(real_train, label_key=args.label_key,
                             stride=args.stride_train, limit_items=args.limit_train or 0)
    val_ds   = DirNPZDataset(real_val,   label_key=args.label_key,
                             stride=args.stride_eval,  limit_items=args.limit_val or 0) if real_val else None
    test_ds  = DirNPZDataset(real_test,  label_key=args.label_key,
                             stride=args.stride_eval,  limit_items=args.limit_test or 0) if real_test else None
    fake_ds  = DirNPZDataset(args.fake_dir, label_key=args.label_key,
                             stride=args.stride_eval,  limit_items=args.limit_fake or 0)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True,
                              num_workers=n_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False,
                              num_workers=n_workers, pin_memory=pin_mem) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False,
                              num_workers=n_workers, pin_memory=pin_mem) if test_ds else None
    fake_loader  = DataLoader(fake_ds,  batch_size=args.batch, shuffle=False, drop_last=False,
                              num_workers=n_workers, pin_memory=pin_mem)

    # Infer channels
    C = int(train_ds[0][0].shape[0])

    # Model
    model = make_model(args.arch, c_in=C, n_cls=n_cls).to(device)

    # Train
    train_model(model, train_loader, device, epochs=args.epochs, lr=args.lr, use_tqdm=args.tqdm)

    out = {"meta": {
                "class_names": names, "arch": args.arch, "label_key": args.label_key,
                "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                "device": device.type,
                "stride_train": args.stride_train, "stride_eval": args.stride_eval,
                "limit_train": args.limit_train, "limit_val": args.limit_val,
                "limit_test": args.limit_test, "limit_fake": args.limit_fake
            }}

    # Baseline (real test)
    if test_loader is not None:
        base = evaluate(model, test_loader, device, use_tqdm=args.tqdm, desc="eval:test")
        out["baseline"] = base

    # Recovery (fake)
    rec = evaluate(model, fake_loader, device, use_tqdm=args.tqdm, desc="eval:fake")
    rec["n_fake"] = int(len(fake_ds))

    # Intended match if augment_artifact provided
    if args.augment_artifact is not None:
        try:
            if str(args.augment_artifact).isdigit():
                tgt_idx = int(args.augment_artifact)
            else:
                tgt_idx = name2idx[str(args.augment_artifact)]
            im, n = intended_match(model, fake_loader, device, tgt_idx, use_tqdm=args.tqdm)
            rec["intended_match"] = float(im)
        except Exception as e:
            rec["intended_match"] = None
            rec["_im_error"] = str(e)
    out["recovery"] = rec

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
