# eval/classifier_eval.py
import os, sys, json, glob, argparse, math, re
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# ---- project constants ----
try:
    from utils.constants import ARTIFACT_SET
except Exception:
    ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]


# ----------------- data loaders -----------------

def _stack_npz_split(npz_dir: str, split: str, limit: int | None = None):
    Xs, Ys, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz"))):
        with np.load(f) as z:
            # data
            x = z["x"] if "x" in z.files else z[z.files[0]]
            # labels (prefer y_artifact, else a, else y)
            y = None
            for k in ("y_artifact","a","y"):
                if k in z.files:
                    y = z[k]
                    break
            if y is None:
                raise RuntimeError(f"No labels found in {f}. Expected one of y_artifact/a/y.")
            Xs.append(x.astype(np.float32))
            Ys.append(y.astype(np.int64))
            n += x.shape[0]
            if limit is not None and n >= limit:
                break
    if not Xs:
        return np.zeros((0,8,1), np.float32), np.zeros((0,), np.int64)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    if limit is not None and X.shape[0] > limit:
        X, Y = X[:limit], Y[:limit]
    return X.astype(np.float32), Y.astype(np.int64)

def _infer_artifact_from_path_or_meta(fake_dir: str) -> int | None:
    # meta.json wins if present
    meta_p = os.path.join(fake_dir, "meta.json")
    if os.path.exists(meta_p):
        try:
            meta = json.load(open(meta_p))
            art = str(meta.get("artifact","")).lower()
            if art in ARTIFACT_SET:
                return ARTIFACT_SET.index(art)
        except Exception:
            pass
    # fallback: look for artifact substring in directory name
    name = os.path.basename(os.path.normpath(fake_dir)).lower()
    for i,a in enumerate(ARTIFACT_SET):
        if re.search(rf"\b{re.escape(a)}\b", name):
            return i
    return None

def _load_fake_dir(fake_dir: str, limit: int | None = None):
    """Load generated windows from samples.npy or *.npz (key 'x')."""
    # samples.npy path
    npy = os.path.join(fake_dir, "samples.npy")
    Xs = []
    if os.path.exists(npy):
        X = np.load(npy).astype(np.float32)
        if limit is not None: X = X[:limit]
        return X
    # shards
    n = 0
    for f in sorted(glob.glob(os.path.join(fake_dir, "*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
            if key is None: continue
            x = z[key].astype(np.float32)
            Xs.append(x)
            n += x.shape[0]
            if limit is not None and n >= limit:
                break
    return (np.concatenate(Xs, axis=0).astype(np.float32) if Xs else
            np.zeros((0,8,1), np.float32))

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


# ----------------- classifier backbones -----------------

class TinyEEG1D(nn.Module):
    """Very small 1D CNN (baseline)."""
    def __init__(self, c_in=8, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)

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

class EEGNetSmall(nn.Module):
    """
    Minimal EEGNet-like classifier (2D ops). Input [B,C,T] -> [B,1,C,T].
    """
    def __init__(self, c_in=8, n_classes=7, F1=8, D=2, F2=16, kernel_len=64, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False)
        self.bn1   = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1*D, (c_in, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1*D)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropout)
        self.separable_depth = nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding=(0,8), bias=False)
        self.separable_point = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        self.bn3       = nn.BatchNorm2d(F2)
        self.pool2     = nn.AvgPool2d((1, 8))
        self.drop2     = nn.Dropout(dropout)
        self.fc        = nn.Linear(F2, n_classes)
    def forward(self, x):  # x: [B,C,T]
        x = x.unsqueeze(1)                  # [B,1,C,T]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.depthwise(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = torch.relu(self.bn3(self.separable_point(self.separable_depth(x))))
        x = self.pool2(x); x = self.drop2(x)
        x = x.mean(dim=(2,3))               # GAP over C,T
        return self.fc(x)


def make_model(arch: str, c_in: int, n_classes: int):
    if arch == "tiny":
        return TinyEEG1D(c_in=c_in, n_classes=n_classes)
    if arch == "resnet1d":
        return ResNet1DTiny(c_in=c_in, n_classes=n_classes)
    if arch == "eegnet":
        return EEGNetSmall(c_in=c_in, n_classes=n_classes)
    raise ValueError(f"Unknown arch: {arch}")


# ----------------- train / eval utils -----------------

def train_classifier(model, Xtr, ytr, Xval, yval, epochs=8, batch=256, lr=1e-3, device=None, class_weight=None, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or pick_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if class_weight is not None:
        cw = torch.tensor(class_weight, dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=cw)
    else:
        crit = nn.CrossEntropyLoss()

    def _batches(X, y, bs, shuffle=True):
        idx = np.arange(len(X))
        if shuffle: np.random.shuffle(idx)
        for i in range(0, len(X), bs):
            j = idx[i:i+bs]
            yield (torch.from_numpy(X[j]).to(device), torch.from_numpy(y[j]).to(device))

    best_val = -1.0
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(_batches(Xtr, ytr, batch, shuffle=True), total=math.ceil(len(Xtr)/batch), desc=f"train ep{ep}/{epochs}", leave=False)
        for xb, yb in pbar:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # quick val
        model.eval()
        with torch.no_grad():
            yv_pred = []
            for xb, yb in _batches(Xval, yval, batch, shuffle=False):
                logits = model(xb)
                yv_pred.append(logits.argmax(dim=1).cpu().numpy())
            yv_pred = np.concatenate(yv_pred) if len(yv_pred) else np.array([])
        f1v = f1_score(yval, yv_pred, average="macro") if len(yv_pred) else 0.0
        best_val = max(best_val, f1v)
    return model

def predict(model, X, batch=256, device=None):
    device = device or pick_device()
    model = model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            logits = model(xb)
            outs.append(logits.softmax(dim=1).cpu().numpy())
    P = np.concatenate(outs, axis=0) if outs else np.zeros((0,1), np.float32)
    yhat = P.argmax(axis=1) if P.size else np.array([], dtype=np.int64)
    return yhat, P


# ----------------- main logic -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", default=None, help="Folder with generated samples for recovery eval.")
    ap.add_argument("--augment_with", default=None, help="Folder with generated samples to add to training for utility eval.")
    ap.add_argument("--fake_artifact", default=None, help="Force intended artifact label for fake_dir (e.g., 'eye').")
    ap.add_argument("--task", default="artifact", choices=["artifact"], help="Currently only artifact classification.")
    ap.add_argument("--arch", type=str, default="tiny", choices=["tiny","resnet1d","eegnet"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--class_weight", type=str, default="none", choices=["none","balanced"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert (args.fake_dir is not None) ^ (args.augment_with is not None), \
        "Provide exactly one of --fake_dir (recovery) or --augment_with (utility)."

    # load real data
    Xtr, ytr = _stack_npz_split(args.real_dir, "train", limit=args.limit_train)
    Xva, yva = _stack_npz_split(args.real_dir, "val",   limit=args.limit_test)   # use val as held-out during training
    Xte, yte = _stack_npz_split(args.real_dir, "test",  limit=args.limit_test)

    n_classes = len(ARTIFACT_SET)
    c_in = Xtr.shape[1]
    device = pick_device()

    # optional class weights
    class_weight = None
    if args.class_weight == "balanced" and len(ytr) > 0:
        counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        class_weight = (inv * (n_classes / inv.sum())).tolist()

    # ----------------- RECOVERY (specificity) -----------------
    if args.fake_dir is not None:
        # train on real (train), validate on real (val)
        model = make_model(args.arch, c_in=c_in, n_classes=n_classes)
        model = train_classifier(model, Xtr, ytr, Xva, yva,
                                 epochs=args.epochs, batch=args.batch, lr=args.lr,
                                 device=device, class_weight=class_weight, seed=args.seed)

        # load fake
        Xf = _load_fake_dir(args.fake_dir, limit=None)
        # intended label (single class) if not embedded
        if args.fake_artifact is not None:
            art_name = args.fake_artifact.lower()
            if art_name not in ARTIFACT_SET:
                raise ValueError(f"--fake_artifact must be one of {ARTIFACT_SET}")
            intended = ARTIFACT_SET.index(art_name)
        else:
            intended = _infer_artifact_from_path_or_meta(args.fake_dir)
        # default to 'none' if we truly cannot infer
        if intended is None: intended = 0
        y_fake = np.full((Xf.shape[0],), intended, dtype=np.int64)

        # predict on fake
        yhat, P = predict(model, Xf, batch=args.batch, device=device)
        acc = accuracy_score(y_fake, yhat) if len(yhat) else 0.0
        f1m = f1_score(y_fake, yhat, average="macro") if len(yhat) else 0.0
        intended_match = float((yhat == intended).mean()) if len(yhat) else 0.0

        # per-class prediction counts (nice for debugging)
        pred_counts = {ARTIFACT_SET[i]: int((yhat==i).sum()) for i in range(n_classes)}

        out = {
            "arch": args.arch,
            "train": {"n": int(len(Xtr)), "epochs": args.epochs, "class_weight": args.class_weight},
            "fake_meta": {
                "dir": args.fake_dir, "n_fake": int(len(Xf)),
                "intended_label": int(intended), "intended_name": ARTIFACT_SET[intended],
            },
            "recovery": {
                "acc": float(acc),
                "macro_f1": float(f1m),
                "intended_match": float(intended_match),
                "n_fake": int(len(Xf)),
                "pred_counts": pred_counts
            }
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out}")
        return

    # ----------------- UTILITY (augmentation) -----------------
    if args.augment_with is not None:
        # 1) baseline
        base_model = make_model(args.arch, c_in=c_in, n_classes=n_classes)
        base_model = train_classifier(base_model, Xtr, ytr, Xva, yva,
                                      epochs=args.epochs, batch=args.batch, lr=args.lr,
                                      device=device, class_weight=class_weight, seed=args.seed)
        yhat_b, _ = predict(base_model, Xte, batch=args.batch, device=device)
        base_acc = accuracy_score(yte, yhat_b) if len(yhat_b) else 0.0
        base_f1  = f1_score(yte, yhat_b, average="macro") if len(yhat_b) else 0.0

        # 2) augmentation: load fakes and append to train
        Xf = _load_fake_dir(args.augment_with, limit=None)
        # try to infer a single artifact label if labels not embedded
        intended = _infer_artifact_from_path_or_meta(args.augment_with)
        if intended is None: intended = 0
        yf = np.full((len(Xf),), intended, dtype=np.int64)

        Xtr_aug = np.concatenate([Xtr, Xf], axis=0) if len(Xf) else Xtr
        ytr_aug = np.concatenate([ytr, yf], axis=0) if len(Xf) else ytr

        aug_model = make_model(args.arch, c_in=c_in, n_classes=n_classes)
        aug_model = train_classifier(aug_model, Xtr_aug, ytr_aug, Xva, yva,
                                     epochs=args.epochs, batch=args.batch, lr=args.lr,
                                     device=device, class_weight=class_weight, seed=args.seed)
        yhat_a, _ = predict(aug_model, Xte, batch=args.batch, device=device)
        aug_acc = accuracy_score(yte, yhat_a) if len(yhat_a) else 0.0
        aug_f1  = f1_score(yte, yhat_a, average="macro") if len(yhat_a) else 0.0

        out = {
            "arch": args.arch,
            "baseline": {"acc": float(base_acc), "macro_f1": float(base_f1), "n_train": int(len(Xtr))},
            "augmentation": {
                "acc": float(aug_acc), "macro_f1": float(aug_f1),
                "delta_acc": float(aug_acc - base_acc),
                "delta_macro_f1": float(aug_f1 - base_f1),
                "n_train_aug": int(len(Xtr_aug)),
                "n_aug_added": int(len(Xf)),
                "aug_source": args.augment_with,
                "aug_label_name": ARTIFACT_SET[int(intended)]
            }
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out}")
        return


if __name__ == "__main__":
    main()
