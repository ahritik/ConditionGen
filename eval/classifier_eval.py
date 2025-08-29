# eval/classifier_eval.py
import os, sys, glob, json, argparse, math, random
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# project constants
from utils.constants import ARTIFACT_SET

# --------------------------- utils ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(pref: str = "auto") -> torch.device:
    pref = pref.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")

def infer_fake_artifact_idx(fake_dir: str) -> Optional[int]:
    """Infer intended artifact from dir name like 'samples_eye08' or '.../electrode_...'
       Returns index in ARTIFACT_SET or None."""
    name = os.path.basename(os.path.normpath(fake_dir)).lower()
    for i, a in enumerate(ARTIFACT_SET):
        if a.lower() in name:
            return i
    # try meta.json (if present)
    meta_p = os.path.join(fake_dir, "meta.json")
    if os.path.exists(meta_p):
        try:
            meta = json.load(open(meta_p))
            a = meta.get("artifact", "").lower()
            if a in [x.lower() for x in ARTIFACT_SET]:
                return [x.lower() for x in ARTIFACT_SET].index(a)
        except Exception:
            pass
    return None

# --------------------------- data ----------------------------

def _iter_npz(npz_dir: str, split: Optional[str] = None):
    patt = f"{split}_*.npz" if split else "*.npz"
    for f in sorted(glob.glob(os.path.join(npz_dir, patt))):
        with np.load(f) as z:
            yield f, z

def _stack_real(npz_dir: str, split: str, limit: Optional[int] = None):
    Xs, yA = [], []
    total = 0
    for _, z in _iter_npz(npz_dir, split=split):
        x = z["x"]                       # [N,C,T]
        a = z["y_artifact"]              # [N]
        Xs.append(x); yA.append(a)
        total += x.shape[0]
        if limit is not None and total >= limit:
            break
    if len(Xs) == 0:
        return np.zeros((0,8,1), np.float32), np.zeros((0,), np.int64)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(yA, axis=0)
    if limit is not None and X.shape[0] > limit:
        X, y = X[:limit], y[:limit]
    return X.astype(np.float32), y.astype(np.int64)

def _stack_fake(fake_dir: str, intended_idx: Optional[int] = None, limit: Optional[int] = None):
    """Loads synthetic windows from fake_dir. Labels are either read (if present)
       or all set to intended_idx."""
    Xs, yA = [], []
    n = 0
    for f in sorted(glob.glob(os.path.join(fake_dir, "*.npz"))):
        with np.load(f) as z:
            x = z["x"]  # expect [N,C,T]
            Xs.append(x)
            n += x.shape[0]
            # if labels exist, use them
            if "y_artifact" in z:
                yA.append(z["y_artifact"])
        if limit is not None and n >= limit:
            break
    if len(Xs) == 0:
        return np.zeros((0,8,1), np.float32), np.zeros((0,), np.int64)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    if len(yA) > 0:
        y = np.concatenate(yA, axis=0).astype(np.int64)
    else:
        if intended_idx is None:
            raise ValueError(
                "Could not infer intended artifact for fake_dir and no labels present. "
                "Pass --fake_artifact to specify (e.g., 'eye')."
            )
        y = np.full(X.shape[0], intended_idx, dtype=np.int64)
    if limit is not None and X.shape[0] > limit:
        X, y = X[:limit], y[:limit]
    return X, y

class NpzDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X; self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])     # [C,T]
        y = torch.tensor(int(self.y[idx]))
        return x, y

# --------------------------- model ---------------------------

class TinyEEG1D(nn.Module):
    """Small 1D CNN for artifact classification."""
    def __init__(self, c_in=8, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):  # x: [B,C,T]
        h = self.net(x).squeeze(-1)           # [B,128]
        return self.fc(h)

# -------------------------- metrics --------------------------

def confusion_matrix(pred: np.ndarray, true: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, t in zip(pred, true):
        cm[int(t), int(p)] += 1
    return cm

def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def accuracy_from_cm(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(1, cm.sum()))

# --------------------------- train/eval -----------------------

def train_epoch(model, loader, device, opt, crit, use_tqdm=True):
    model.train()
    it = tqdm(loader, desc="Train", dynamic_ncols=True) if use_tqdm else loader
    run = 0.0
    for i, (xb, yb) in enumerate(it):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        run = 0.98 * run + 0.02 * loss.item() if i else loss.item()
        if use_tqdm: it.set_postfix(loss=f"{run:.4f}")

def evaluate(model, loader, device, n_classes, use_tqdm=True) -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    it = tqdm(loader, desc="Eval", dynamic_ncols=True) if use_tqdm else loader
    with torch.no_grad():
        for xb, yb in it:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            preds.append(pred); trues.append(yb.numpy())
    if len(trues) == 0:
        return {"acc": 0.0, "macro_f1": 0.0}
    y = np.concatenate(trues); p = np.concatenate(preds)
    cm = confusion_matrix(p, y, n_classes)
    return {
        "acc": accuracy_from_cm(cm),
        "macro_f1": macro_f1_from_cm(cm),
        "cm": cm.tolist()
    }

# ----------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="NPZ dir with real splits (train/val/test)")
    ap.add_argument("--fake_dir", default=None, help="Dir with synthetic NPZs (for recovery eval)")
    ap.add_argument("--augment_with", default=None, help="Dir with synthetic NPZs to augment *training* for utility eval")
    ap.add_argument("--task", default="artifact", choices=["artifact"], help="Supported: artifact")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_train", type=int, default=None, help="subsample train windows")
    ap.add_argument("--limit_test", type=int, default=None, help="subsample test windows")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_tqdm", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake_artifact", type=str, default=None,
                    help="Intended artifact label for fake_dir (e.g., 'eye'); used if labels absent and inference fails.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    use_tqdm = not args.no_tqdm

    n_classes = len(ARTIFACT_SET)
    # ---------------- Baseline classifier (real train -> real test) ----------------
    Xtr, ytr = _stack_real(args.real_dir, split="train", limit=args.limit_train)
    Xte, yte = _stack_real(args.real_dir, split="test", limit=args.limit_test)
    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise SystemExit("Empty dataset: check --real_dir contents and/or split files.")

    ds_tr = NpzDataset(Xtr, ytr)
    ds_te = NpzDataset(Xte, yte)
    train_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyEEG1D(c_in=Xtr.shape[1], n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        if use_tqdm: print(f"\nEpoch {ep+1}/{args.epochs} (baseline)")
        train_epoch(model, train_loader, device, opt, crit, use_tqdm=use_tqdm)

    base_metrics = evaluate(model, test_loader, device, n_classes, use_tqdm=use_tqdm)
    result = {
        "mode": "baseline",
        "task": args.task,
        "baseline": {
            "acc": base_metrics["acc"],
            "macro_f1": base_metrics["macro_f1"],
            "confusion_matrix": base_metrics["cm"],
            "classes": ARTIFACT_SET,
            "n_train": int(len(ds_tr)),
            "n_test": int(len(ds_te)),
        }
    }

    # ---------------- Specificity / recovery on fake_dir ----------------
    if args.fake_dir:
        intended_idx = infer_fake_artifact_idx(args.fake_dir)
        if args.fake_artifact is not None:
            # override if user specified
            if args.fake_artifact.lower() in [x.lower() for x in ARTIFACT_SET]:
                intended_idx = [x.lower() for x in ARTIFACT_SET].index(args.fake_artifact.lower())
            else:
                raise SystemExit(f"--fake_artifact must be one of {ARTIFACT_SET}")
        Xf, yf = _stack_fake(args.fake_dir, intended_idx=intended_idx, limit=args.limit_test)
        ds_fake = NpzDataset(Xf, yf)
        fake_loader = DataLoader(ds_fake, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

        spec_metrics = evaluate(model, fake_loader, device, n_classes, use_tqdm=use_tqdm)
        # also compute "intended match" rate (when intended_idx is uniform)
        intended_match = None
        if intended_idx is not None and (len(set(yf.tolist())) == 1):
            # how often the classifier predicts the intended artifact
            preds = []
            model.eval()
            with torch.no_grad():
                for xb, _ in fake_loader:
                    xb = xb.to(device)
                    preds.append(model(xb).argmax(1).cpu().numpy())
            p = np.concatenate(preds)
            intended_match = float((p == intended_idx).mean())

        result["recovery"] = {
            "fake_dir": args.fake_dir,
            "acc": spec_metrics["acc"],
            "macro_f1": spec_metrics["macro_f1"],
            "confusion_matrix": spec_metrics["cm"],
            "intended_match": intended_match,
            "intended_label": (ARTIFACT_SET[intended_idx] if intended_idx is not None else None),
            "n_fake": int(len(ds_fake))
        }

    # ---------------- Utility / augmentation ----------------
    if args.augment_with:
        aug_intended = infer_fake_artifact_idx(args.augment_with)
        if args.fake_artifact is not None:
            if args.fake_artifact.lower() in [x.lower() for x in ARTIFACT_SET]:
                aug_intended = [x.lower() for x in ARTIFACT_SET].index(args.fake_artifact.lower())
            else:
                raise SystemExit(f"--fake_artifact must be one of {ARTIFACT_SET}")
        Xa, ya = _stack_fake(args.augment_with, intended_idx=aug_intended, limit=args.limit_train)
        # concat to real train
        X_aug = np.concatenate([Xtr, Xa], axis=0)
        y_aug = np.concatenate([ytr, ya], axis=0)
        ds_aug = NpzDataset(X_aug, y_aug)
        aug_loader = DataLoader(ds_aug, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        model_aug = TinyEEG1D(c_in=Xtr.shape[1], n_classes=n_classes).to(device)
        opt_aug = torch.optim.AdamW(model_aug.parameters(), lr=args.lr)
        crit_aug = nn.CrossEntropyLoss()

        for ep in range(args.epochs):
            if use_tqdm: print(f"\nEpoch {ep+1}/{args.epochs} (augmented)")
            train_epoch(model_aug, aug_loader, device, opt_aug, crit_aug, use_tqdm=use_tqdm)

        aug_metrics = evaluate(model_aug, test_loader, device, n_classes, use_tqdm=use_tqdm)
        result["augmentation"] = {
            "augment_with": args.augment_with,
            "acc": aug_metrics["acc"],
            "macro_f1": aug_metrics["macro_f1"],
            "confusion_matrix": aug_metrics["cm"],
            "n_train_aug": int(len(ds_aug)),
        }
        # deltas
        result["augmentation"]["delta_acc"] = float(aug_metrics["acc"] - base_metrics["acc"])
        result["augmentation"]["delta_macro_f1"] = float(aug_metrics["macro_f1"] - base_metrics["macro_f1"])

    # ---------------- save ----------------
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()
