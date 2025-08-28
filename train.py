# train.py
import os, argparse, time, csv, math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data.loaders_tuar_tusz import make_loader
from models.unet1d_film import UNet1DFiLM
from models.conditioning import ConditionEmbed
from models.diffusion import Diffusion, EMA
from utils.constants import ARTIFACT_SET

def autocast_maybe(device_type="cpu", enabled=True):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    if device_type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    elif device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        return nullcontext()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to training NPZ created by make_windows.py")
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--log_tb", action="store_true", help="Enable TensorBoard logging")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    ap.add_argument("--widths", type=str, default="64,128,256")
    ap.add_argument("--stft_lambda", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    loader = make_loader(args.npz, batch_size=args.batch_size, shuffle=True, num_workers=0 if device=="mps" else 2, pin_memory=False)
    widths = tuple(int(x) for x in args.widths.split(","))

    # Build model + cond embed + diffusion
    film_provider = ConditionEmbed(film_dim=widths[0])
    net = UNet1DFiLM(in_channels=loader.dataset.x.shape[1], widths=widths, num_res_blocks=2, film_provider=film_provider).to(device)
    diff = Diffusion(net, T=1000, stft_lambda=args.stft_lambda, device=device).to(device)

    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr)
    ema = EMA(diff, decay=0.999)

    # Logging
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "train_log.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if os.stat(csv_path).st_size == 0:
        csv_w.writerow(["step","loss","loss_mse","loss_stft"])
    writer = SummaryWriter(args.out_dir) if args.log_tb else None

    start_step = 0
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        diff.load_state_dict(ckpt["diff"])
        opt.load_state_dict(ckpt["opt"])
        ema.shadow = ckpt["ema"]
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))  # CUDA scaler; MPS uses autocast without scaler

    step = start_step
    net.train()
    while step < args.steps:
        for x, cd in loader:
            step += 1
            x = x.to(device)
            # normalize cond shapes
            cond = {
                "artifact": cd["artifact"].to(device),
                "intensity": cd["intensity"].to(device).view(-1,1),
                "seizure": cd["seizure"].to(device).view(-1,1),
                "age_bin": cd["age_bin"].to(device),
                "montage_id": cd["montage_id"].to(device),
            }

            opt.zero_grad(set_to_none=True)
            ctx = autocast_maybe(device, enabled=(device in ["cuda","mps"]))
            with ctx:
                loss, logs = diff(x, cond)
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            ema.update(diff)

            # Logging
            if step % 50 == 0:
                csv_w.writerow([step, float(loss.detach().cpu()), float(logs["loss_mse"]), float(logs["loss_stft"])])
                csv_f.flush()
                if writer:
                    writer.add_scalar("train/loss", float(loss.detach().cpu()), global_step=step)
                    writer.add_scalar("train/loss_mse", float(logs["loss_mse"]), global_step=step)
                    writer.add_scalar("train/loss_stft", float(logs["loss_stft"]), global_step=step)

            if step % args.ckpt_every == 0 or step == args.steps:
                ck = {
                    "diff": diff.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.shadow,
                    "step": step
                }
                p = os.path.join(args.out_dir, f"ckpt_{step}.pt")
                torch.save(ck, p)
                print(f"[CKPT] Saved {p}")

            if step >= args.steps:
                break

    csv_f.close()
    if writer:
        writer.flush(); writer.close()

if __name__ == "__main__":
    main()
