#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample.py
---------
Sampler that matches the 6-class TUAR taxonomy (no movement).

- Loads class order from the training label_map.json next to the checkpoint.
- Builds the same 12-D conditioning vector as training (no intensity).
- Works with classifier-free guidance and DDIM/Heun samplers.
"""

from __future__ import annotations
import os, json, argparse
import numpy as np
from tqdm import tqdm

import torch
torch.set_float32_matmul_precision("high")

# Robust imports (with or without a 'models' package)
try:
    from models.unet1d_film import UNet1DFiLM
    from models.diffusion import Diffusion
except ImportError:
    from unet1d_film import UNet1DFiLM
    from diffusion import Diffusion

from conditioning import load_label_map_from, build_cond_torch, ARTIFACTS_CANON

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda"), "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"

def load_ckpt(ckpt_path: str, model: torch.nn.Module):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    if isinstance(state, dict) and "model" in state and "ema_model" in state["model"]:
        state = state["model"]
    model.load_state_dict(state if isinstance(state, dict) else ckpt["model"], strict=False)
    return ckpt.get("step", None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to training checkpoint (.pt)")
    ap.add_argument("--artifact", required=True, type=str, help="artifact name (must be in label_map.json)")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shape", type=int, nargs=2, default=[8, 800], help="C T")
    ap.add_argument("--seizure", type=int, default=0)
    ap.add_argument("--age_bin", type=int, default=1)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--sampler", choices=["ddim", "heun"], default="ddim")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Class order (should be 6 for TUAR)
    CLASS_NAMES = load_label_map_from(args.ckpt) or ARTIFACTS_CANON
    NAME2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
    n_artifacts = len(CLASS_NAMES)
    if args.artifact not in NAME2IDX:
        raise SystemExit(f"--artifact '{args.artifact}' not found in classes: {CLASS_NAMES}")

    # Device + model
    device, dev = pick_device()
    C, T = int(args.shape[0]), int(args.shape[1])
    cond_dim = n_artifacts + 6  # 6: seizure(1) + age(4) + montage(1)
    net = UNet1DFiLM(channels=C, widths=(64, 128, 256), cond_dim=cond_dim).to(device)
    model = Diffusion(net, T=1000).to(device)
    step = load_ckpt(args.ckpt, model)
    print(f"[ckpt] loaded step={step} on {dev}")

    # Conditioning vector (12-D for TUAR)
    art_idx = NAME2IDX[args.artifact]
    cond_single = build_cond_torch(
        artifact_idx=art_idx,
        seizure=args.seizure,
        age_bin=args.age_bin,
        montage_id=args.montage_id,
        n_artifacts=n_artifacts,
        device=device,
    )
    cond = cond_single.unsqueeze(0).repeat(args.n, 1)  # [n,cond_dim]

    # Sample
    with torch.no_grad():
        if args.sampler == "ddim":
            x = model.ddim_sample(n=args.n, cond=cond, steps=args.steps, guidance=args.guidance,
                                  eta=args.eta, batch=args.batch, shape=(C, T), device=device)
        else:
            x = model.heun_sample(n=args.n, cond=cond, steps=args.steps, guidance=args.guidance,
                                  batch=args.batch, shape=(C, T), device=device)
    x = x.cpu().numpy()

    # Save
    base = os.path.join(args.out_dir, f"synth_{args.artifact}")
    if args.save_npz:
        np.savez_compressed(base + ".npz", x=x, artifact=np.full((args.n,), art_idx, dtype=np.int32),
                            seizure=args.seizure, age_bin=args.age_bin, montage_id=args.montage_id)
    else:
        np.save(base + ".npy", x)

    meta = {
        "ckpt": args.ckpt, "step": int(step) if step is not None else None,
        "class_names": CLASS_NAMES, "artifact": args.artifact, "artifact_idx": int(art_idx),
        "n": int(args.n), "steps": int(args.steps), "guidance": float(args.guidance),
        "eta": float(args.eta), "batch": int(args.batch), "shape": [C, T],
        "seizure": int(args.seizure), "age_bin": int(args.age_bin), "montage_id": int(args.montage_id),
        "sampler": args.sampler,
    }
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[write] {base}.(npy|npz) and {base}.json")

if __name__ == "__main__":
    main()
