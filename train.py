#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
--------
Training script for ConditionGen on TUAR NPZ shards (6-class taxonomy; no movement).

- models.conditioning: 12-D = 6(one-hot artifact) + 1(seizure) + 4(one-hot age) + 1(montage scalar)
- Class order read from dataset's label_map.json; a copy is saved to --log_dir
- Supports NEW shard keys (artifact, seizure, age_bin, montage_id) and LEGACY (y_*)
- tqdm progress bars + TensorBoard scalars (optional)

Usage:
  python train.py \
    --npz_dir out/tuar_npz/train --val_npz_dir out/tuar_npz/val \
    --log_dir out/runs/condgen_tuar_6cls --epochs 50 --batch 256 --lr 2e-4 --timesteps 1000
"""
from __future__ import annotations
import os, glob, csv, json, argparse, random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

# Robust imports (with or without a 'models' package)
try:
    from models.unet1d_film import UNet1DFiLM
except ImportError:
    from unet1d_film import UNet1DFiLM

try:
    from models.diffusion import Diffusion, EMA
except ImportError:
    from diffusion import Diffusion, EMA

from models.conditioning import save_label_map, load_label_map_from, build_cond_np, ARTIFACTS_CANON


# ------------------------------- Dataset -------------------------------------

class TUARDataset(Dataset):
    """
    Minimal TUAR NPZ dataset loader.

    Accepts NPZ shards with either:
      NEW: x, artifact, seizure, age_bin, montage_id, intensity
      OLD: x, y_artifact, y_seizure, y_agebin, y_montage, intensity
    """
    def __init__(self, root: str, class_names: List[str], shuffle_files: bool = True):
        super().__init__()
        self.root = root
        self.class_names = class_names
        self.name2idx = {n: i for i, n in enumerate(class_names)}

        self.files = sorted(glob.glob(os.path.join(root, "*.npz")))
        if shuffle_files:
            random.shuffle(self.files)
        if not self.files:
            raise RuntimeError(f"No .npz shards found under {root}")

        self._index = []  # list of (file_idx, row_idx)
        for fi, f in enumerate(self.files):
            with np.load(f) as npz:
                n = npz["x"].shape[0]
            self._index.extend([(fi, i) for i in range(n)])

        self._cache = None  # (fi, npz_obj)

    def __len__(self):
        return len(self._index)

    def _open(self, fi: int):
        return np.load(self.files[fi])

    def __getitem__(self, idx: int):
        fi, row = self._index[idx]
        if self._cache is None or self._cache[0] != fi:
            if self._cache is not None:
                try:
                    self._cache[1].close()
                except Exception:
                    pass
            self._cache = (fi, self._open(fi))
        z = self._cache[1]

        x = z["x"][row].astype(np.float32)  # [C,T]

        def pick(*keys, default=None):
            for k in keys:
                if k in z:
                    return z[k][row]
            if default is None:
                raise KeyError(f"None of keys {keys} found in shard")
            return default

        a = pick("artifact", "y_artifact")
        if isinstance(a, (np.str_, str, bytes)):  # string labels rare
            a = self.name2idx[str(a)]
        else:
            a = int(a)

        s = int(pick("seizure", "y_seizure"))
        g = int(pick("age_bin", "y_agebin"))
        m = int(pick("montage_id", "y_montage"))

        cond = build_cond_np(
            artifact_idx=a, seizure=s, age_bin=g, montage_id=m, n_artifacts=len(self.class_names)
        ).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(cond)


def make_loader(root: str, class_names: List[str], batch: int, shuffle: bool,
                pin_memory: bool, num_workers: int = 4) -> DataLoader:
    ds = TUARDataset(root, class_names, shuffle_files=shuffle)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True,
                      num_workers=num_workers, pin_memory=pin_memory)


# ------------------------------- Utilities -----------------------------------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def save_ckpt(path: str, step: int, net: nn.Module, opt: torch.optim.Optimizer, ema):
    os.makedirs(os.path.dirname(path), exist_ok=True, mode=0o755)
    state = {"step": step, "model": net.state_dict(), "opt": opt.state_dict()}
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


# --------------------------------- Main --------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True, help="training NPZ folder")
    ap.add_argument("--val_npz_dir", default=None, help="optional validation NPZ folder")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--stft_win", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--lambda_stft", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--tb_every", type=int, default=10, help="log scalars to TensorBoard every N steps (0=off)")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # 1) Class order from dataset label_map.json (TUAR: 6 classes)
    class_names = load_label_map_from(args.npz_dir) or ARTIFACTS_CANON
    n_artifacts = len(class_names)
    lm_path = save_label_map(args.log_dir, class_names)
    print(f"[meta] wrote {lm_path} with {n_artifacts} classes: {class_names}")

    # 2) Device
    device, devtype = pick_device()
    print(f"[device] {devtype}")

    # pin memory only helps (and is supported) on CUDA
    pin_mem = (device.type == "cuda")

    # 3) Data — also infer channels C safely from one sample
    tmp_ds = TUARDataset(args.npz_dir, class_names, shuffle_files=False)
    C = int(tmp_ds[0][0].shape[0])
    del tmp_ds

    train_loader = make_loader(args.npz_dir, class_names, args.batch, shuffle=True,
                               pin_memory=pin_mem, num_workers=4)
    val_loader = None
    if args.val_npz_dir and os.path.isdir(args.val_npz_dir):
        val_loader = make_loader(args.val_npz_dir, class_names, args.batch, shuffle=False,
                                 pin_memory=pin_mem, num_workers=2)

    # 4) Model (UNet with FiLM expecting cond_dim = n_artifacts + 6 -> 12 for TUAR)
    cond_dim = n_artifacts + 6
    # NOTE: match your UNet1DFiLM signature (c_in / c_hidden), not channels/widths
    net = UNet1DFiLM(c_in=C, c_hidden=(64, 128, 256), cond_dim=cond_dim).to(device)

    # 5) Diffusion wrapper (v-pred + optional STFT-L1)
    model = Diffusion(
        net, T=args.timesteps, stft_win=args.stft_win, stft_hop=args.stft_hop, lambda_stft=args.lambda_stft
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))
    ema = EMA(model, decay=args.ema_decay)

    # 6) Logging: CSV + optional TensorBoard
    csv_path = os.path.join(args.log_dir, "train_log.csv")
    tb_dir = os.path.join(args.log_dir, "tb")
    writer = None
    if SummaryWriter is not None and args.tb_every > 0:
        try:
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"[tb] logging to {tb_dir} (every {args.tb_every} steps)")
        except Exception as e:  # pragma: no cover
            print(f"[tb] disabled ({e})")

    with open(csv_path, "w", newline="") as csv_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["epoch", "step", "loss", "base", "stft_l1", "snr_mean", "lr"])

        # 7) Train
        step = 0
        model.train()
        for epoch in range(1, args.epochs + 1):
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True, leave=False)
            for x, c in pbar:
                x = x.to(device)            # [B,C,T]
                c = c.to(device)            # [B,cond_dim]

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(not args.no_amp and device.type == "cuda")):
                    loss, parts = model(x, c)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                ema.update(model)

                base = float(parts.get("base", 0.0))
                stft = float(parts.get("stft_l1", 0.0))
                snr  = float(parts.get("snr_mean", 0.0))
                lval = float(loss.detach().cpu())
                lr   = float(opt.param_groups[0]["lr"])

                # CSV log every 50 steps
                if step % 50 == 0:
                    csv_w.writerow([epoch, step, lval, base, stft, snr, lr])

                # TensorBoard log (lightweight cadence)
                if writer is not None and args.tb_every > 0 and (step % args.tb_every == 0):
                    writer.add_scalar("loss/total", lval, step)
                    writer.add_scalar("loss/base",  base, step)
                    writer.add_scalar("loss/stft_l1", stft, step)
                    writer.add_scalar("snr/mean", snr, step)
                    writer.add_scalar("opt/lr", lr, step)

                # tqdm postfix occasionally
                if step % 10 == 0:
                    pbar.set_postfix(loss=lval, base=base, stft=stft)

                # Periodic checkpoints
                if step % 1000 == 0 and step > 0:
                    ckdir = os.path.join(args.log_dir, "checkpoints")
                    save_ckpt(os.path.join(ckdir, f"step_{step}.pt"), step, model, opt, ema)
                    save_ckpt(os.path.join(ckdir, "last.pt"), step, model, opt, ema)

                step += 1

            # Epoch-end checkpoint
            ckdir = os.path.join(args.log_dir, "checkpoints")
            save_ckpt(os.path.join(ckdir, f"epoch_{epoch}.pt"), step, model, opt, ema)
            save_ckpt(os.path.join(ckdir, "last.pt"), step, model, opt, ema)
            print(f"[epoch {epoch}] last loss={lval:.4f}")

    if writer is not None:
        writer.close()

    print(f"[done] wrote log to {csv_path}")
    if writer is not None:
        print(f"[tb] view with: tensorboard --logdir \"{tb_dir}\" --port 6006")


if __name__ == "__main__":
    main()
