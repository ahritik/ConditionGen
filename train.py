import os, csv, math, argparse, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.loaders_tuar_tusz import NPZShardDataset
from models.unet1d_film import UNet1DFiLM
from models.diffusion import Diffusion1D, EMA
from utils.constants import ARTIFACT_SET

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_ckpt(path, model, opt, ema, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "step": step,
        "args": vars(args)
    }
    torch.save(ckpt, path)

def load_ckpt(path, model, opt=None, ema=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt"):
        opt.load_state_dict(ckpt["opt"])
    if ema is not None and ckpt.get("ema"):
        # Load EMA into current EMA object
        for s, tgt in zip(ckpt["ema"], ema.shadow):
            tgt.copy_(s)
    return ckpt.get("step", 0), ckpt.get("args", {})

def collate(batch):
    # Pad/truncate to the same T (should already be identical)
    x = torch.stack([b["x"] for b in batch], dim=0)  # [B,C,T]
    # Build cond vector expected by UNet: here we directly use provided cond_vec
    # Additionally, we pass intensity as part of cond vector (last dim could be montage scalar; leave as is)
    cond_vecs = []
    for b in batch:
        cond_vecs.append(b["cond_vec"])
    cond = torch.stack(cond_vecs, dim=0)
    return x, cond

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_stft", type=float, default=0.1)
    ap.add_argument("--stft_win", type=int, default=128)
    ap.add_argument("--stft_hop", type=int, default=64)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_tb", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    device = get_device()
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "train_log.csv")
    tb = SummaryWriter(args.log_dir) if args.log_tb else None

    train_ds = NPZShardDataset(args.npz_dir, split="train")
    val_ds   = NPZShardDataset(args.npz_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate, drop_last=False)

    cond_dim = (len(ARTIFACT_SET) + 1 + 4 + 1)
    net = UNet1DFiLM(c_in=8, c_hidden=(64,128,256), cond_dim=cond_dim)
    model = Diffusion1D(net, timesteps=1000, stft_lambda=args.lambda_stft, stft_cfg=(args.stft_win, args.stft_hop, args.stft_win))

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, beta=0.999)

    start_step = 0
    if args.resume:
        start_step, _ = load_ckpt(args.resume, model, opt, ema)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    use_mps = (device.type == "mps")
    amp_enabled = use_mps or torch.cuda.is_available()

    train_iter = iter(train_loader)
    with open(csv_path, "a", newline="") as cf:
        cw = csv.writer(cf)
        if start_step == 0:
            cw.writerow(["step","loss","mse","stft"])
        for step in tqdm(range(start_step, args.steps), total=args.steps-start_step, desc="Train"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x, cond = batch
            x = x.to(device)
            cond = cond.to(device)

            t = torch.randint(0, model.timesteps, (x.size(0),), device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss, parts = model(x, cond, t)

            opt.zero_grad(set_to_none=True)
            if amp_enabled and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            ema.update(model)

            if step % 50 == 0:
                row = [step, float(loss.item()), parts.get("mse", float("nan")), parts.get("stft", float("nan"))]
                cw.writerow(row)
                cf.flush()
                if tb:
                    tb.add_scalar("loss/total", loss.item(), step)
                    for k,v in parts.items():
                        tb.add_scalar(f"loss/{k}", v, step)

            if (step > 0) and (step % args.ckpt_every == 0):
                ckpt_path = os.path.join(args.log_dir, f"checkpoints/step_{step:06d}.pt")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                save_ckpt(ckpt_path, model, opt, ema, step, args)
                # EMA checkpoint
                tmp = [p.detach().clone() for p in model.parameters()]
                ema.copy_to(model)
                ckpt_path_ema = os.path.join(args.log_dir, f"checkpoints/step_{step:06d}_ema.pt")
                save_ckpt(ckpt_path_ema, model, opt, ema, step, args)
                # restore params
                for p, tparam in zip(model.parameters(), tmp):
                    p.data.copy_(tparam)

    if tb:
        tb.close()

if __name__ == "__main__":
    main()
