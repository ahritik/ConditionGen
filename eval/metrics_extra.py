# eval/metrics_extra.py
import os, json, glob, argparse, math, re
import numpy as np

from scipy.signal import welch
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn

try:
    from utils.constants import BANDS as _BANDS, ARTIFACT_SET
except Exception:
    _BANDS = {"delta":(0.5,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30)}
    ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

# ---------------- IO ----------------

def _stack_npz_split(npz_dir: str, split: str, limit: int | None = None, with_labels: bool = False):
    Xs, Ys, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
            if key is None: continue
            x = z[key].astype(np.float32)
            Xs.append(x); n += x.shape[0]
            if with_labels:
                y = None
                for k in ("y_artifact","a","y"):
                    if k in z.files: y = z[k].astype(np.int64); break
                if y is None: raise RuntimeError(f"No labels in {f}")
                Ys.append(y)
            if limit is not None and n >= limit: break
    if not Xs:
        return (np.zeros((0,8,1), np.float32), np.zeros((0,), np.int64)) if with_labels else np.zeros((0,8,1), np.float32)
    X = np.concatenate(Xs, axis=0)
    if limit is not None and X.shape[0] > limit: X = X[:limit]
    if with_labels:
        Y = np.concatenate(Ys, axis=0)
        if limit is not None and Y.shape[0] > limit: Y = Y[:limit]
        return X, Y
    return X

def _load_fake_dir(fake_dir: str, limit: int | None = None):
    npy = os.path.join(fake_dir, "samples.npy")
    if os.path.exists(npy):
        X = np.load(npy).astype(np.float32)
        return X if limit is None else X[:limit]
    Xs, n = [], 0
    for f in sorted(glob.glob(os.path.join(fake_dir, "*.npz"))):
        with np.load(f) as z:
            key = "x" if "x" in z.files else (z.files[0] if len(z.files) else None)
            if key is None: continue
            x = z[key].astype(np.float32)
            Xs.append(x); n += x.shape[0]
            if limit is not None and n >= limit: break
    return np.concatenate(Xs, axis=0) if Xs else np.zeros((0,8,1), np.float32)

def _infer_artifact_from_dir(fake_dir: str):
    meta_p = os.path.join(fake_dir, "meta.json")
    if os.path.exists(meta_p):
        try:
            art = json.load(open(meta_p)).get("artifact","").lower()
            if art in ARTIFACT_SET: return art
        except Exception:
            pass
    name = os.path.basename(os.path.normpath(fake_dir)).lower()
    for a in ARTIFACT_SET:
        if re.search(rf"\b{re.escape(a)}\b", name):
            return a
    return "unknown"

# ---------------- Feature helpers ----------------

def bandpower_features(X, fs):
    N,C,T = X.shape
    nperseg = min(512, T)
    XC = X.reshape(N*C, T)
    freqs, Pxx = welch(XC, fs=fs, nperseg=nperseg)
    feats = []
    for _,(lo,hi) in _BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        bp = Pxx[:, m].mean(axis=1)
        feats.append(np.log10(bp + 1e-12))
    F = np.stack(feats, axis=1)      # [N*C, 4]
    return F.reshape(N, C*len(_BANDS)).astype(np.float32)  # [N, 32]

def hjorth_params(x):
    x = x.astype(np.float64)
    dx = np.diff(x)
    var_x = np.var(x) + 1e-12
    var_dx = np.var(dx) + 1e-12
    mob = math.sqrt(var_dx / var_x)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx) + 1e-12
    mob_dx = math.sqrt(var_ddx / var_dx)
    comp = mob_dx / (mob + 1e-12)
    return mob, comp, var_x

def hjorth_feats(X):
    N,C,T = X.shape
    out = np.zeros((N, C*3), dtype=np.float32)
    for i in range(N):
        arr = []
        for c in range(C):
            mob, comp, varx = hjorth_params(X[i,c])
            arr += [mob, comp, np.log(varx + 1e-12)]
        out[i] = np.array(arr, dtype=np.float32)
    return out  # [N, 24]

def cov_uppertri(X):
    N,C,T = X.shape
    ut_idx = np.triu_indices(C)
    out = np.zeros((N, len(ut_idx[0])), dtype=np.float32)
    for i in range(N):
        cov = np.cov(X[i], rowvar=True)
        out[i] = cov[ut_idx]
    return out  # [N, 36]

# ---------------- Tiny ResNet for clf features ----------------

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
    def forward_features(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return x  # [B, widths[2]]

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def train_resnet_features(Xtr, ytr, Xva, yva, epochs=8, batch=256, lr=1e-3, seed=1234):
    torch.manual_seed(seed); np.random.seed(seed)
    device = pick_device()
    n_classes = len(ARTIFACT_SET)
    model = ResNet1DTiny(c_in=Xtr.shape[1], n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    def _batches(X, y, bs, shuffle=True):
        idx = np.arange(len(X))
        if shuffle: np.random.shuffle(idx)
        for i in range(0, len(X), bs):
            j = idx[i:i+bs]
            yield torch.from_numpy(X[j]).to(device), torch.from_numpy(y[j]).to(device)

    model.train()
    for ep in range(epochs):
        for xb, yb in _batches(Xtr, ytr, batch, True):
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    return model.to(device)

def embed_with_resnet(model, X, batch=512):
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            outs.append(model.forward_features(xb).cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,128), np.float32)

# ---------------- Metrics ----------------

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def compute_ffd(F_real, F_fake):
    m_r = F_real.mean(0); m_f = F_fake.mean(0)
    c_r = np.cov(F_real, rowvar=False)
    c_f = np.cov(F_fake, rowvar=False)
    return frechet_distance(m_r, c_r, m_f, c_f)

def compute_mmd_rbf(F_real, F_fake, subsample_for_bandwidth=2000):
    rng = np.random.default_rng(123)
    A, B = F_real, F_fake
    idx_a = rng.choice(len(A), size=min(subsample_for_bandwidth, len(A)), replace=False)
    idx_b = rng.choice(len(B), size=min(subsample_for_bandwidth, len(B)), replace=False)
    samp = np.vstack([A[idx_a], B[idx_b]])
    dists = pairwise_distances(samp, samp, metric="euclidean")
    med = np.median(dists); sigma = med if med > 1e-12 else 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    Krr = rbf_kernel(A, A, gamma=gamma)
    Kff = rbf_kernel(B, B, gamma=gamma)
    Krf = rbf_kernel(A, B, gamma=gamma)
    m, n = len(A), len(B)
    mmd2 = Krr.sum()/(m*m) + Kff.sum()/(n*n) - 2.0 * Krf.sum()/(m*n)
    return float(mmd2)

def knn_precision_recall(F_real, F_fake, k=3):
    if len(F_real) == 0 or len(F_fake) == 0: return 0.0, 0.0
    nn_real = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(F_real)
    d_r, idx_r = nn_real.kneighbors(F_real, n_neighbors=k+1, return_distance=True)
    rad_r = d_r[:, -1]
    d_fr, i_fr = nn_real.kneighbors(F_fake, n_neighbors=1, return_distance=True)
    precision = float(np.mean(d_fr[:,0] <= rad_r[i_fr[:,0]])) if len(F_fake) else 0.0
    nn_fake = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(F_fake)
    d_f, idx_f = nn_fake.kneighbors(F_fake, n_neighbors=k+1, return_distance=True)
    rad_f = d_f[:, -1]
    d_rf, i_rf = nn_fake.kneighbors(F_real, n_neighbors=1, return_distance=True)
    recall = float(np.mean(d_rf[:,0] <= rad_f[i_rf[:,0]])) if len(F_real) else 0.0
    return precision, recall

def one_nn_two_sample_acc(F_real, F_fake, folds=5, k=5, seed=1234):
    X = np.vstack([F_real, F_fake]); y = np.array([0]*len(F_real) + [1]*len(F_fake))
    mu = X.mean(0); sd = X.std(0) + 1e-12
    Xn = (X - mu)/sd
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(Xn, y):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(Xn[tr], y[tr])
        accs.append(clf.score(Xn[te], y[te]))
    return float(np.mean(accs))

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--feature_kind", type=str, default="stat92",
                    choices=["stat92","spec32","clf_resnet"])
    ap.add_argument("--no_window_norm", action="store_true",
                    help="Disable extra per-window z-score inside metrics.")
    ap.add_argument("--clf_epochs", type=int, default=8)
    ap.add_argument("--clf_batch", type=int, default=256)
    ap.add_argument("--clf_lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # real/fake windows
    Xr = _stack_npz_split(args.real_dir, args.split, limit=args.limit, with_labels=False)
    Xf = _load_fake_dir(args.fake_dir, limit=args.limit)
    n = min(len(Xr), len(Xf))
    if n == 0:
        out = dict(error="empty inputs", n_real=len(Xr), n_fake=len(Xf))
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"Wrote {args.out} (empty inputs)")
        return
    idx_r = np.random.choice(len(Xr), size=n, replace=False) if len(Xr) > n else np.arange(len(Xr))
    idx_f = np.random.choice(len(Xf), size=n, replace=False) if len(Xf) > n else np.arange(len(Xf))
    Xr = Xr[idx_r]; Xf = Xf[idx_f]

    # optional per-window normalization inside the metric
    if not args.no_window_norm:
        Xr = (Xr - Xr.mean(axis=2, keepdims=True)) / (Xr.std(axis=2, keepdims=True) + 1e-6)
        Xf = (Xf - Xf.mean(axis=2, keepdims=True)) / (Xf.std(axis=2, keepdims=True) + 1e-6)

    # feature extraction
    if args.feature_kind == "spec32":
        Fr = bandpower_features(Xr, fs=args.fs)
        Ff = bandpower_features(Xf, fs=args.fs)
    elif args.feature_kind == "clf_resnet":
        # train on real train+val, embed both sets
        Xtr, ytr = _stack_npz_split(args.real_dir, "train", limit=None, with_labels=True)
        Xva, yva = _stack_npz_split(args.real_dir, "val",   limit=None, with_labels=True)
        if not args.no_window_norm:
            for arr in (Xtr, Xva):
                arr -= arr.mean(axis=2, keepdims=True); arr /= (arr.std(axis=2, keepdims=True) + 1e-6)
        model = train_resnet_features(Xtr, ytr, Xva, yva,
                                      epochs=args.clf_epochs, batch=args.clf_batch, lr=args.clf_lr, seed=args.seed)
        Fr = embed_with_resnet(model, Xr, batch=512)
        Ff = embed_with_resnet(model, Xf, batch=512)
    else:  # stat92 (default)
        Fr = np.concatenate([bandpower_features(Xr, args.fs), hjorth_feats(Xr), cov_uppertri(Xr)], axis=1)
        Ff = np.concatenate([bandpower_features(Xf, args.fs), hjorth_feats(Xf), cov_uppertri(Xf)], axis=1)

    # standardize jointly for stability
    mu = np.vstack([Fr, Ff]).mean(0); sd = np.vstack([Fr, Ff]).std(0) + 1e-12
    Frn = (Fr - mu)/sd; Ffn = (Ff - mu)/sd

    ffd  = compute_ffd(Frn, Ffn)
    mmd  = compute_mmd_rbf(Frn, Ffn)
    prec, rec = knn_precision_recall(Frn, Ffn, k=args.k)
    two_sample_acc = one_nn_two_sample_acc(Frn, Ffn, folds=5, k=5, seed=args.seed)

    out = dict(
        artifact=_infer_artifact_from_dir(args.fake_dir),
        n_real=int(len(Fr)), n_fake=int(len(Ff)), feature_dim=int(Fr.shape[1]),
        feature_kind=args.feature_kind, window_norm=(not args.no_window_norm),
        ffd=float(ffd), mmd_rbf=float(mmd),
        knn_precision=float(prec), knn_recall=float(rec),
        nn_two_sample_acc=float(two_sample_acc)
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
