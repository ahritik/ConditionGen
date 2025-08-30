# train.py
# Training script for ConditionGen diffusion model (TUAR 8-ch EEG windows)
from __future__ import annotations
import os, glob, csv, json, math, argparse, random
from typing import List, Tuple
import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.unet1d_film import UNet1DFiLM
from models.diffusion import Diffusion1D as Diffusion, EMA

# ---- constants (keep in sync with utils/constants.py) ----
ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]
N_CH = 8
T_SAMPLES = 800  # 4s @ 200Hz


# ---------------- data ----------------

class NPZWindows(Dataset):
    """
    Loads a list of NPZ files (e.g., train_*.npz). Each file has arrays:
        x: [N, C=8, T=800] float32
        a: [N] artifact idx in [0..6]
        s: [N] seizure flag {0,1} (optional -> zeros)
        g: [N] age bin idx in [0..3] (optional -> zeros)
        m: [N] montage id idx (optional -> zeros)
    """
    def __init__(self, npz_dir: str, split: str):
        paths = sorted(glob.glob(os.path.join(npz_dir, f"{split}_*.npz")))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found: {npz_dir}/{split}_*.npz")
        Xs, As, Ss, Gs, Ms = [], [], [], [], []
        for p in paths:
            with np.load(p) as z:
                key = "x" if "x" in z.files else z.files[0]
                x = z[key].astype(np.float32)
                a = (z["a"] if "a" in z.files else z.get("y_artifact", np.zeros(len(x), np.int64))).astype(np.int64)
                s = (z["s"] if "s" in z.files else np.zeros(len(x), np.int64)).astype(np.int64)
                g = (z["g"] if "g" in z.files else np.zeros(len(x), np.int64)).astype(np.int64)
                m = (z["m"] if "m" in z.files else np.zeros(len(x), np.int64)).astype(np.int64)
                Xs.append(x); As.append(a); Ss.append(s); Gs.append(g); Ms.append(m)
        self.x = np.concatenate(Xs, 0)
        self.a = np.concatenate(As, 0)
        self.s = np.concatenate(Ss, 0)
        self.g = np.concatenate(Gs, 0)
        self.m = np.concatenate(Ms, 0)

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, idx: int):
        x = self.x[idx]
        a = int(self.a[idx]); s = int(self.s[idx]); g = int(self.g[idx]); m = int(self.m[idx])
        cond = build_cond(a, s, g, m)  # 13-dim
        return torch.from_numpy(x), torch.from_numpy(cond)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v

def build_cond(artifact_idx: int, seizure: int, age_bin: int, montage_id: int) -> np.ndarray:
    """
    13-D condition vector to match checkpoints:
      [artifact_onehot(7)] + [seizure(1)] + [age_onehot(4)] + [montage_id_scalar(1)]
    (No intensity term here; intensity is only used at sampling time.)
    """
    a = one_hot(artifact_idx, 7)
    s = np.array([float(seizure)], dtype=np.float32)
    g = one_hot(age_bin, 4)
    m = np.array([float(montage_id)], dtype=np.float32)
    return np.concatenate([a, s, g, m], axis=0).astype(np.float32)

# ------------ make model ---------------
def make_unet(n_ch=8, widths=(64,128,256), cond_dim=13):
    import inspect
    sig = inspect.signature(UNet1DFiLM.__init__)
    names = list(sig.parameters.keys())
    # try common arg names for "input channels"
    for k in ("in_channels","in_chans","in_ch","ch_in","channels","C_in","c_in","n_channels"):
        if k in names:
            kwargs = {k: n_ch}
            if "widths" in names: kwargs["widths"] = widths
            if "cond_dim" in names: kwargs["cond_dim"] = cond_dim
            return UNet1DFiLM(**kwargs)
    # last resort: call with only supported args
    kwargs = {}
    if "widths" in names: kwargs["widths"] = widths
    if "cond_dim" in names: kwargs["cond_dim"] = cond_dim
    return UNet1DFiLM(**kwargs)

# ---------------- utils ----------------

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"

class NullScaler:
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): return False

def save_ckpt(path, step, model, opt, ema):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
    }, path)
    print(f"[save] {path} (step={step})")

def load_ckpt(path, model, opt=None, ema=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    step = int(ckpt.get("step", 0))

    # locate model state_dict
    cand_keys = ["model", "state_dict", "net", "weights"]
    sd = None
    for k in cand_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            sd = ckpt[k]; break
    if sd is None:
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            raise RuntimeError("No model state_dict found in checkpoint")

    # filter to current model
    msd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape}
    missing = [k for k in msd.keys() if k not in filtered]
    unexpected = [k for k in sd.keys() if k not in msd]

    model.load_state_dict(filtered, strict=False)
    print(f"[load] model: loaded {len(filtered)}/{len(msd)} tensors "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")

    if opt is not None:
        for k in ("opt", "optimizer", "opt_state"):
            if k in ckpt:
                try:
                    opt.load_state_dict(ckpt[k])
                    print("[load] optimizer: ok")
                except Exception as e:
                    print(f"[load] optimizer: skipped ({e})")
                break

    if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
        try:
            ema.load_state_dict(ckpt["ema"])
            print("[load] ema: ok")
        except Exception as e:
            maybe = ckpt["ema"]
            if isinstance(maybe, dict) and all(isinstance(v, torch.Tensor) for v in maybe.values()):
                try:
                    for name, ten in maybe.items():
                        if name in ema.shadow:
                            ema.shadow[name] = ten.clone()
                    print("[load] ema: ok (shadow fallback)")
                except Exception as e2:
                    print(f"[load] ema: skipped ({e2})")
            else:
                print(f"[load] ema: skipped ({e})")

    return step, ckpt


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200000, help="target GLOBAL step (not epochs)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stft_win", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--lambda_stft", type=float, default=0.1)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_tb", action="store_true")
    ap.add_argument("--no_amp", action="store_true", help="disable AMP (use fp32)")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device, devtype = pick_device()
    print(f"[device] {devtype}")

    # model
    net = make_unet(n_ch=N_CH, widths=(64,128,256), cond_dim=13).to(device)
    net.to(device)

    model = Diffusion(
        net, T=1000,
        stft_win=args.stft_win, stft_hop=args.stft_hop,
        lambda_stft=args.lambda_stft, snr_clip=5.0, schedule="cosine"
    ).to(device)

    # optimizer + EMA
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(net, beta=0.999)  # tracks the UNet weights
    ema.to(device)

    # data
    ds_train = NPZWindows(args.npz_dir, "train")
    ds_val   = NPZWindows(args.npz_dir, "val")
    dl = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    it = iter(dl)

    # AMP
    use_amp = (not args.no_amp) and (devtype in ("cuda", "mps"))
    if devtype == "cuda" and use_amp:
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
    elif devtype == "mps" and use_amp:
        # MPS fp16 is fast but can produce NaNs; leave enabled only if stable
        scaler = NullScaler()
        autocast_ctx = torch.amp.autocast("mps", dtype=torch.float16)
    else:
        # pure fp32 (safe mode)
        scaler = NullScaler()
        autocast_ctx = contextlib.nullcontext()

    # resume
    start_step = 0
    if args.resume:
        start_step, _ = load_ckpt(args.resume, model, opt, ema, device=device)
        print(f"[resume] Loaded checkpoint at global step {start_step}")

    # logging
    csv_path = os.path.join(args.log_dir, "train_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if os.path.getsize(csv_path) == 0:
        csv_w.writerow(["step","loss_total","loss_base","loss_stft","lr"])

    tb = None
    if args.log_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(args.log_dir)
        except Exception as e:
            print(f"[tb] failed to init tensorboard: {e}")

    # helper to sample a small val batch and log PSD-ish proxy (optional)
    def log_parts(step, loss, parts):
        csv_w.writerow([step, float(loss), parts.get("base",0.0), parts.get("stft_l1",0.0), args.lr])
        csv_f.flush()
        if tb:
            tb.add_scalar("loss/total", float(loss), step)
            tb.add_scalar("loss/base",  parts.get("base",0.0), step)
            tb.add_scalar("loss/stft",  parts.get("stft_l1",0.0), step)
            tb.add_scalar("misc/snr_mean", parts.get("snr_mean",0.0), step)

    # train loop
    model.train()
    step = start_step
    from tqdm import trange
    pbar = trange(start_step, args.steps, initial=start_step, total=args.steps, desc="Train")

    while step < args.steps:
        try:
            x, cond = next(it)
        except StopIteration:
            dl = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
            it = iter(dl)
            x, cond = next(it)

        x = x.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        with autocast_ctx:
            t = torch.randint(0, model.timesteps, (x.size(0),), device=device, dtype=torch.long)
            loss, parts = model(x, cond, t)
        
            # --- NaN/Inf guard ---
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {step}; skipping and reducing LR")
                for pg in opt.param_groups:
                    pg["lr"] = pg["lr"] * 0.5
                # zero grad & re-init batch
                opt.zero_grad(set_to_none=True)
                # optional: turn off AMP if it was on
                use_amp = False
                # skip update
                step += 1
                continue


        opt.zero_grad(set_to_none=True)
        if isinstance(scaler, NullScaler):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        # EMA update on the UNet (not the wrapper)
        ema.update(net)

        step += 1
        pbar.update(1)
        if step % 50 == 0:
            log_parts(step, loss.item(), parts)

        if step % args.ckpt_every == 0 or step == args.steps:
            ckdir = os.path.join(args.log_dir, "checkpoints")
            os.makedirs(ckdir, exist_ok=True)
            # standard step checkpoint
            save_ckpt(os.path.join(ckdir, f"step_{step}.pt"), step, model, opt, ema)
            # update last.pt
            save_ckpt(os.path.join(ckdir, "last.pt"), step, model, opt, ema)
            # EMA export for sampling
            # (copy EMA -> net -> wrap into a "model" dict with wrapper schedule as well)
            # Users will typically sample with sample.py which already handles EMA.
            pass

    csv_f.close()
    if tb: tb.close()


if __name__ == "__main__":
    main()
