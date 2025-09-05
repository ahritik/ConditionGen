#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifier_eval.py
------------------
6-class TUAR recovery/augmentation evaluator (no movement).

- Trains a small 1D CNN on real shards (class order from label_map.json).
- Reports baseline metrics on real test.
- Optionally evaluates fakes in --fake_dir and computes Intended-Match (IM).
"""

from __future__ import annotations
import os, json, glob, argparse, random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from conditioning import load_label_map_from, ARTIFACTS_CANON

# ------------------------------ Data -----------------------------------------

class NPZClassifierDS(Dataset):
    def __init__(self, root: str, class_names: List[str], shuffle_files: bool = True):
        super().__init__()
        self.root = root
        self.class_names = class_names
        self.name2idx = {n: i for i, n in enumerate(class_names)}
        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if shuffle_files: random.shuffle(self.files)
        if not self.files: raise RuntimeError(f"No npz files in {root}")
        self.index = []
        for fi, fp in enumerate(self.files):
            with np.load(fp, allow_pickle=True) as npz:
                n = npz["x"].shape[0]
            self.index += [(fi, i) for i in range(n)]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int):
        fi, row = self.index[idx]
        fp = self.files[fi]
        with np.load(fp, allow_pickle=True) as npz:
            x = npz["x"][row].astype(np.float32)  # [C,T]
            y = npz["artifact"][row] if "artifact" in npz else npz["y_artifact"][row]
            if isinstance(y, (np.integer, int)):
                y_idx = int(y)
            else:
                y_idx = self.name2idx[str(y)]
        return torch.tensor(x), torch.tensor(y_idx, dtype=torch.long)

def make_loader(root: str, class_names: List[str], batch: int, shuffle: bool) -> DataLoader:
    ds = NPZClassifierDS(root, class_names, shuffle_files=shuffle)
    def _collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False,
                      num_workers=4, pin_memory=True, collate_fn=_collate)

# ------------------------------ Model ----------------------------------------

class SmallResNet1D(nn.Module):
    def __init__(self, in_ch: int, n_cls: int):
        super().__init__()
        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv1d(cin, cout, 7, stride=stride, padding=3),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Conv1d(cout, cout, 3, padding=1),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
            )
        self.stem = nn.Sequential(nn.Conv1d(in_ch, 64, 7, padding=3),
                                  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.layer1 = block(64, 128, stride=2)
        self.layer2 = block(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, n_cls)

    def forward(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

# ------------------------------ Metrics --------------------------------------

@torch.no_grad()
def eval_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    n_correct = 0; n_total = 0
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)

    for x, y in loader:
        x = x.to(device); y = y.to(device)
        pred = model(x).argmax(dim=1)
        n_correct += (pred == y).sum().item()
        n_total += y.numel()
        for t, p in zip(y.view(-1).tolist(), pred.view(-1).tolist()):
            if p == t: tp[t] += 1
            else: fp[p] += 1; fn[t] += 1

    acc = n_correct / max(1, n_total)
    n_cls = max(1, max(list(tp.keys()) + list(fp.keys()) + list(fn.keys()) + [0]) + 1)
    f1s = []
    for k in range(n_cls):
        precision = tp[k] / max(1, tp[k] + fp[k])
        recall = tp[k] / max(1, tp[k] + fn[k])
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return {"acc": float(acc), "macro_f1": float(np.mean(f1s) if f1s else 0.0)}

def intended_match(pred_counts: Dict[str, int], target_name: str) -> float:
    total = sum(pred_counts.values()) or 1
    return pred_counts.get(target_name, 0) / total

# --------------------------------- Main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_train", required=True)
    ap.add_argument("--real_val", required=False, default=None)
    ap.add_argument("--real_test", required=True)
    ap.add_argument("--fake_dir", required=False, default=None)
    ap.add_argument("--augment_artifact", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Class order from dataset; fallback to TUAR 6-class if absent
    CLASS_NAMES = load_label_map_from(args.real_train, args.real_test) or ARTIFACTS_CANON
    NAME2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
    print(f"[meta] class order: {CLASS_NAMES}")

    # Data
    train_loader = make_loader(args.real_train, CLASS_NAMES, args.batch, shuffle=True)
    val_loader = make_loader(args.real_val, CLASS_NAMES, args.batch, shuffle=False) if args.real_val else None
    test_loader = make_loader(args.real_test, CLASS_NAMES, args.batch, shuffle=False)

    # Model / train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = next(iter(train_loader))[0].shape[1]
    model = SmallResNet1D(in_ch=C, n_cls=len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            ce(model(x), y).backward()
            opt.step()
        if val_loader:
            m = eval_classifier(model, val_loader, device)
            print(f"[epoch {epoch}] val acc={m['acc']:.3f} macro_f1={m['macro_f1']:.3f}")

    baseline = eval_classifier(model, test_loader, device)
    out = {"baseline": baseline}
    print(f"[baseline] test acc={baseline['acc']:.3f} macro_f1={baseline['macro_f1']:.3f}")

    # Recovery on fakes (optional)
    if args.fake_dir:
        pred_counts = {name: 0 for name in CLASS_NAMES}

        xs = []
        for fp in glob.glob(os.path.join(args.fake_dir, "*.npz")):
            with np.load(fp, allow_pickle=True) as npz:
                xs.append(npz["x"].astype(np.float32))
        if not xs:
            for fp in glob.glob(os.path.join(args.fake_dir, "*.npy")):
                xs.append(np.load(fp).astype(np.float32))

        if xs:
            X = torch.tensor(np.concatenate(xs, axis=0))
            model.eval()
            with torch.no_grad():
                for i0 in tqdm(range(0, X.shape[0], args.batch), desc="recovery"):
                    x = X[i0:i0+args.batch].to(device)
                    pred = model(x).argmax(dim=1).cpu().numpy().tolist()
                    for p in pred:
                        pred_counts[CLASS_NAMES[p]] += 1

        target_name = None
        for jf in glob.glob(os.path.join(args.fake_dir, "*.json")):
            try:
                j = json.load(open(jf))
                if "artifact" in j: target_name = str(j["artifact"]); break
            except Exception:
                pass

        im = intended_match(pred_counts, target_name) if target_name else None
        out["recovery"] = {
            "pred_counts": pred_counts,
            "intended": target_name,
            "intended_match": float(im) if im is not None else None,
            "n_fake": int(sum(pred_counts.values())),
        }
        print(f"[recovery] IM={out['recovery']['intended_match']} for '{target_name}'")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {args.out}")

if __name__ == "__main__":
    main()
