#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust sampler for ConditionGen.
- Works with base (EMA) and fine-tuned (no-EMA) checkpoints
- Handles different UNet1DFiLM constructor arg names (c_in / channels / in_ch)
- Uses Diffusion.ddim_sample with correct kwargs (cond=(B,cond_dim), guidance, shape=(C,T))
- Chunked sampling with tqdm
"""

import os, json, argparse, time, inspect
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ----------------- imports -----------------
def _try_import():
    try:
        from models.unet1d_film import UNet1DFiLM
        from models.diffusion import Diffusion
        return UNet1DFiLM, Diffusion
    except Exception:
        from unet1d_film import UNet1DFiLM  # type: ignore
        from diffusion import Diffusion      # type: ignore
        return UNet1DFiLM, Diffusion

UNet1DFiLM, Diffusion = _try_import()

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]
ART2IDX = {a:i for i,a in enumerate(ARTIFACT_SET)}
C, T = 8, 800  # EEG shape

# --------------- condition builder (13 dims) ---------------
def build_cond_vec(artifact:str, intensity:float, seizure:int, age_bin:int, montage_id:int) -> torch.Tensor:
    """
    Layout (matches training):
      [ one-hot artifact (7) | intensity (1) | age (4 one-hot) | montage_id (1 scalar) ]  = 13
    """
    if artifact not in ART2IDX:
        raise ValueError(f"Unknown artifact: {artifact}. Choices: {ARTIFACT_SET}")
    a = torch.zeros(len(ARTIFACT_SET), dtype=torch.float32)
    a[ART2IDX[artifact]] = 1.0

    inten = torch.tensor([float(intensity)], dtype=torch.float32)

    age = torch.zeros(4, dtype=torch.float32)
    age_idx = max(0, min(3, int(age_bin)))
    age[age_idx] = 1.0

    mont = torch.tensor([float(montage_id)], dtype=torch.float32)  # scalar feature (not one-hot)

    return torch.cat([a, inten, age, mont], dim=0)  # (13,)

# --------------- builders ----------------
def make_unet(c_in:int, widths:tuple, cond_dim:int, device:torch.device):
    sig = inspect.signature(UNet1DFiLM.__init__)
    arg_names = list(sig.parameters.keys())
    kwargs = {}
    for k in ("c_in","channels","in_ch","in_channels"):
        if k in arg_names:
            kwargs[k] = c_in
            break
    for k in ("c_hidden","widths","ch_mult","base_channels"):
        if k in arg_names:
            kwargs[k] = tuple(widths)
            break
    if "cond_dim" in arg_names:
        kwargs["cond_dim"] = cond_dim
    return UNet1DFiLM(**kwargs).to(device)

def make_diffusion(net, T:int, device:torch.device):
    return Diffusion(net, T=T).to(device)

# --------------- ckpt loader ---------------
def _is_diffusion_state(sd:dict)->bool:
    if not isinstance(sd, dict): return False
    for k in sd.keys():
        if k in ("betas","alphas_cumprod","alphas_cumprod_prev","sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod","c0","c1"):
            return True
        if isinstance(k, str) and k.startswith("model."):
            return True
    return False

def load_ckpt_into_diffuser(diffuser:Diffusion, ckpt_path:str, use_ema:bool=True)->tuple[int, list, list]:
    sd = torch.load(ckpt_path, map_location="cpu")
    step = int(sd.get("step", 0)) if isinstance(sd, dict) else 0

    cand = None
    if isinstance(sd, dict):
        if use_ema and ("ema" in sd and isinstance(sd["ema"], dict)):
            cand = sd["ema"]
        elif "model" in sd and isinstance(sd["model"], dict):
            cand = sd["model"]
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            cand = sd["state_dict"]
        elif _is_diffusion_state(sd):
            cand = sd
    if cand is None:
        cand = sd

    needs_prefix = isinstance(cand, dict) and cand and all(isinstance(k,str) and not k.startswith("model.") for k in cand.keys())
    if needs_prefix and any(isinstance(k,str) and "." in k for k in cand.keys()):
        cand = {f"model.{k}": v for k,v in cand.items()}

    missing, unexpected = diffuser.load_state_dict(cand, strict=False)
    return step, missing, unexpected

# --------------- sampling (chunked + batched cond) ---------------
@torch.no_grad()
def sample_all(diffuser:Diffusion, cond_base:torch.Tensor, n:int, steps:int, guidance:float, chunk:int,
               device:torch.device)->np.ndarray:
    """
    Calls diffusion.ddim_sample repeatedly in chunks.
    Ensures cond has shape (B,cond_dim) for FiLM (was the crash).
    """
    cond_base = cond_base.to(device).float()  # (13,)
    out = []
    pbar = tqdm(total=n, desc="Sampling", unit="sig")
    done = 0
    while done < n:
        bs = min(chunk, n - done)
        cond = cond_base.unsqueeze(0).repeat(bs, 1)  # (bs, cond_dim) <-- critical fix
        x = diffuser.ddim_sample(n=bs, cond=cond, steps=steps, guidance=guidance,
                                 shape=(C, T), device=device)
        out.append(x.detach().cpu().numpy())
        done += bs
        pbar.update(bs)
    pbar.close()
    return np.concatenate(out, axis=0)  # (n, C, T)

# --------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint (ema or last)")
    ap.add_argument("--use_ema", action="store_true", help="Load EMA weights if available")
    ap.add_argument("--artifact", type=str, default="none", choices=ARTIFACT_SET)
    ap.add_argument("--intensity", type=float, default=0.6)
    ap.add_argument("--seizure", type=int, default=0)  # kept for compat
    ap.add_argument("--age_bin", type=int, default=1)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256, help="chunk size for sampling")
    ap.add_argument("--cond_dim", type=int, default=13)
    ap.add_argument("--widths", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--save_npy", action="store_true")

    args = ap.parse_args()

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[sample] device={device.type}")

    # net + diffusion
    net = make_unet(C, tuple(args.widths), args.cond_dim, device)
    diff = make_diffusion(net, T=1000, device=device)

    # load ckpt
    step, missing, unexpected = load_ckpt_into_diffuser(diff, args.ckpt, use_ema=args.use_ema)
    print(f"[sample] Loaded ckpt step={step}; missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("[sample]   unexpected (first 8):", sorted(list(unexpected))[:8])
    if missing:
        print("[sample]   missing (first 8):", sorted(list(missing))[:8])

    # condition
    cond_vec = build_cond_vec(args.artifact, args.intensity, args.seizure, args.age_bin, args.montage_id)

    # out dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sample
    t0 = time.time()
    X = sample_all(diff, cond_vec, args.n, args.steps, args.guidance, args.batch, device)
    dt = time.time() - t0
    print(f"[sample] Done. Generated {len(X)} signals in {dt/60:.1f} min.")

    # save
    npy_path = out_dir / "samples.npy"
    np.save(npy_path, X.astype(np.float32))
    meta = {
        "ckpt": args.ckpt,
        "use_ema": bool(args.use_ema),
        "artifact": args.artifact,
        "intensity": float(args.intensity),
        "age_bin": int(args.age_bin),
        "montage_id": int(args.montage_id),
        "n": int(args.n),
        "steps": int(args.steps),
        "guidance": float(args.guidance),
        "batch": int(args.batch),
        "cond_dim": int(args.cond_dim),
        "widths": list(map(int, args.widths)),
        "device": device.type,
        "step_ckpt": int(step),
        "shape": [C,T],
    }
    json.dump(meta, open(out_dir/"meta.json","w"), indent=2)
    print(f"[sample] Wrote {npy_path} and meta.json")

if __name__ == "__main__":
    main()
