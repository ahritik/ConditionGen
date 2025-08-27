#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import torch
from models.unet1d_film import UNet1D
from models.conditioning import CondEmbedding
from models.diffusion import DDPM

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=2.0)
    ap.add_argument("--artifact", type=str, default="eye")
    ap.add_argument("--intensity", type=str, default="mid", choices=["low","mid","high"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--T", type=int, default=800)  # 4s @ 200Hz
    ap.add_argument("--mps", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    dev = torch.device("cpu")
    if torch.cuda.is_available(): dev = torch.device("cuda")
    elif args.mps and torch.backends.mps.is_available(): dev = torch.device("mps")

    cond_dim = 128
    net = UNet1D(channels=args.C, widths=(64,128,256), resblocks=2, time_dim=256, cond_dim=cond_dim).to(dev)
    cond_embed = CondEmbedding(artifact_k=len(ARTIFACT_SET), intensity_dim=1, d_model=cond_dim).to(dev)
    ddpm = DDPM(net, timesteps=1000, schedule="cosine").to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev)
    net.load_state_dict(ckpt["net"])
    cond_embed.load_state_dict(ckpt["cond"])
    net.eval(); cond_embed.eval()

    n = args.n
    art_idx = ARTIFACT_SET.index(args.artifact.lower())
    if args.intensity=="low": inten=0.2
    elif args.intensity=="mid": inten=0.5
    else: inten=0.8

    y = torch.full((n,), art_idx, dtype=torch.long, device=dev)
    i = torch.full((n,1), float(inten), dtype=torch.float32, device=dev)
    cond = cond_embed(y, i)

    X = ddpm.ddim_sample((n, args.C, args.T), cond, steps=args.ddim_steps, cfg_scale=args.cfg_scale)
    X = X.detach().cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, artifact=args.artifact, intensity=args.intensity)
    print("Saved:", args.out, "X:", X.shape)

if __name__ == "__main__":
    main()
