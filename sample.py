# sample.py — ConditionGen EEG sampler (DDIM / Heun2) with EMA + tqdm
from __future__ import annotations
import os, json, argparse
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm

import torch
from models.diffusion import Diffusion
from models.unet1d_film import UNet1DFiLM

ARTIFACT_SET = ["none","eye","muscle","chewing","shiver","electrode","movement"]
CANON_SHAPE = (8, 800)  # (C, T) 8ch x 4s@200Hz

# ---------- helpers ----------
def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32); v[i] = 1.0; return v

def build_cond_vec(artifact: str, seizure: int, age_bin: int, montage_id: int) -> np.ndarray:
    a_idx = ARTIFACT_SET.index(artifact)
    a = one_hot(a_idx, 7)                      # 7
    s = np.array([float(seizure)], np.float32) # 1
    g = one_hot(age_bin, 4)                    # 4
    m = np.array([float(montage_id)], np.float32)  # 1
    return np.concatenate([a, s, g, m], axis=0).astype(np.float32)  # 13-dim

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps"), "mps"
    if torch.cuda.is_available():         return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"

def make_unet(n_ch=8, widths=(64,128,256), cond_dim=13):
    """Be tolerant of constructor arg names across repo versions."""
    import inspect
    sig = inspect.signature(UNet1DFiLM.__init__)
    names = list(sig.parameters.keys())
    for k in ("in_channels","in_chans","in_ch","ch_in","channels","C_in","c_in","n_channels"):
        if k in names:
            kwargs = {k: n_ch}
            if "widths" in names: kwargs["widths"] = tuple(widths)
            if "cond_dim" in names: kwargs["cond_dim"] = cond_dim
            return UNet1DFiLM(**kwargs)
    kwargs = {}
    if "widths" in names: kwargs["widths"] = tuple(widths)
    if "cond_dim" in names: kwargs["cond_dim"] = cond_dim
    return UNet1DFiLM(**kwargs)

def _infer_T_from_ckpt_blob(ckpt: Dict) -> int:
    # Look inside wrapper’s buffers if present
    T = 1000
    blob = ckpt.get("model", {})
    candidates = []
    if isinstance(blob, dict):
        for k in ("alphas_cumprod","sqrt_alphas_cumprod","c0"):
            v = blob.get(k, None)
            if isinstance(v, torch.Tensor):
                candidates.append(v.numel())
    if candidates:
        T = int(candidates[0])
    return T

def load_unet_from_ckpt(net: torch.nn.Module, ckpt: Dict, use_ema: bool) -> Tuple[int, float]:
    """Load UNet weights (prefer EMA). Return (T, coverage%)."""
    T = _infer_T_from_ckpt_blob(ckpt)
    # choose source state dict
    sd = None
    if use_ema and "ema" in ckpt and ckpt["ema"] is not None:
        maybe = ckpt["ema"]
        if isinstance(maybe, dict) and "shadow" in maybe and isinstance(maybe["shadow"], dict):
            sd = maybe["shadow"]
        elif isinstance(maybe, dict) and all(isinstance(v, torch.Tensor) for v in maybe.values()):
            sd = maybe
    if sd is None:
        sd = ckpt.get("model", {})
        if not isinstance(sd, dict): sd = ckpt  # very old checkpoints

    msd = net.state_dict()
    filt = {k:v for k,v in sd.items() if k in msd and isinstance(v, torch.Tensor) and msd[k].shape==v.shape}
    net.load_state_dict(filt, strict=False)
    coverage = 100.0 * len(filt) / max(1, len(msd))

    unexpected = [k for k in sd.keys() if k not in msd]
    if unexpected:
        # Hide the known diffusion buffers to keep logs clean
        show = [u for u in unexpected if not any(x in u for x in (
            "alphas_cumprod","sqrt_alphas_cumprod","sqrt_one_minus_alphas_cumprod","c0","c1"
        ))]
        if show:
            print(f"[sample] Note: unexpected keys: {show[:8]}{' ...' if len(show)>8 else ''}")
        else:
            print(f"[sample] Note: unexpected diffusion buffers; OK.")

    print(f"[sample] Loaded ~{coverage:.1f}% of UNet tensors; cond_dim={getattr(net,'cond_dim','?')}, widths={getattr(net,'widths','?')}, T={T}")
    return T, coverage

def stats_report(x: np.ndarray) -> str:
    # x: [N,C,T]
    gmean, gstd = float(x.mean()), float(x.std())
    chstd = x.std(axis=(0,2))
    return f"shape={tuple(x.shape)} dtype={x.dtype} | mean={gmean:.5f} std={gstd:.5f} | per-ch std: {np.round(chstd,4).tolist()}"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="checkpoint with model/ema")
    ap.add_argument("--use_ema", action="store_true", help="use EMA weights if present")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--artifact", type=str, default="none", choices=ARTIFACT_SET)
    ap.add_argument("--intensity", type=float, default=0.5)  # metadata only
    ap.add_argument("--seizure", type=int, default=0)
    ap.add_argument("--age_bin", type=int, default=1)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--save_npy", action="store_true")
    ap.add_argument("--cond_dim", type=int, default=13)
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim","heun2"])
    ap.add_argument("--widths", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--T", type=int, default=0, help="override diffusion timesteps (0=auto)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device, devtype = pick_device()

    # 1) UNet + weights
    net = make_unet(n_ch=CANON_SHAPE[0], widths=args.widths, cond_dim=args.cond_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    T_auto, _ = load_unet_from_ckpt(net, ckpt, use_ema=args.use_ema)
    net.eval()

    T = (args.T if args.T and args.T > 0 else T_auto or 1000)

    # 2) Diffusion wrapper
    diff = Diffusion(net, T=T, stft_win=128, stft_hop=64, lambda_stft=0.0, schedule="cosine").to(device)
    diff.eval()

    # 3) Condition matrix
    cond1 = build_cond_vec(args.artifact, args.seizure, args.age_bin, args.montage_id)  # (13,)
    cond = torch.from_numpy(np.repeat(cond1[None, :], args.n, axis=0)).to(device)

    # 4) Batched sampling with tqdm
    chunks: List[np.ndarray] = []
    pbar = tqdm(total=args.n, desc=f"Sampling {args.artifact}", ncols=88)
    remain = args.n
    ofs = 0
    with torch.no_grad(), torch.amp.autocast(
        device_type=("cuda" if devtype=="cuda" else ("mps" if devtype=="mps" else "cpu")),
        enabled=(devtype!="cpu"),
        dtype=(torch.float16 if devtype!="cpu" else torch.bfloat16)
    ):
        while remain > 0:
            bs = min(args.batch, remain)
            cond_b = cond[ofs:ofs+bs]
            if args.sampler == "ddim":
                xb = diff.ddim_sample(
                    n=bs, cond=cond_b, steps=args.steps, guidance=args.guidance,
                    eta=0.0, batch=bs, shape=CANON_SHAPE, device=device
                )
            else:
                xb = diff.heun2_sample(
                    n=bs, cond=cond_b, steps=args.steps, guidance=args.guidance,
                    batch=bs, shape=CANON_SHAPE, device=device
                )
            chunks.append(xb.detach().cpu().float().numpy())
            ofs += bs
            remain -= bs
            pbar.update(bs)
    pbar.close()

    x = np.concatenate(chunks, axis=0)  # [N,C,T]

    # 5) Save + meta + quick stats
    meta = {
        "artifact": args.artifact,
        "intensity": float(args.intensity),
        "seizure": int(args.seizure),
        "age_bin": int(args.age_bin),
        "montage_id": int(args.montage_id),
        "n": int(args.n),
        "steps": int(args.steps),
        "guidance": float(args.guidance),
        "sampler": args.sampler,
        "shape": list(CANON_SHAPE),
        "cond_dim": int(args.cond_dim),
        "widths": list(args.widths),
        "dev": devtype,
        "ckpt": args.ckpt,
        "used_ema": bool(args.use_ema),
        "T": int(T),
        "seed": int(args.seed),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if args.save_npy:
        np.save(os.path.join(args.out_dir, "samples.npy"), x)

    print(f"[sample] {stats_report(x)}")
    print(f"[sample] Wrote {args.out_dir}  (N={x.shape[0]}, C={x.shape[1]}, T={x.shape[2]})")

if __name__ == "__main__":
    main()
