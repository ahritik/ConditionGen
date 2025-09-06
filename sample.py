#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample.py — Conditional EEG diffusion sampler (TUAR, 6 artifacts)

- Uses your repo sampler: models.diffusion.Diffusion(...).ddim_sample(...) or heun2_sample(...)
- Adds a progress-enabled local sampler: --sampler ddim_local (with --tqdm)
- Builds 12-D cond vec internally: [6 one-hot artifact, 1 seizure, 4 one-hot age, 1 montage_id]
- --cond_scale multiplies cond vec before FiLM (stronger conditioning)
- Guidance handled inside sampler (pseudo-CFG via zero-cond)

Robust bits:
- UNet state-dict finder (ema.shadow / net / model / state_dict / flat)
- Local SimpleDDIM fallback (eta=0) with tqdm for steps/batches
"""

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.unet1d_film import UNet1DFiLM
from models.diffusion import Diffusion as ModelDiffusion


# ---------------- utils ----------------

def device_pick():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_artifact_names_fallback():
    for p in ("out/tuar_npz/label_map.json", "out/npz/label_map.json"):
        if os.path.exists(p):
            try:
                return list(json.load(open(p))["artifact_names"])
            except Exception:
                pass
    return ["none","eye","muscle","chewing","shiver","electrode"]


def build_cond_numpy(n, artifact, seizure, age_bin, montage_id, canon=None, expect_dim=12):
    names = list(canon) if canon else _load_artifact_names_fallback()
    K = len(names)
    if isinstance(artifact, str):
        art_idx = names.index(artifact)
    else:
        art_idx = int(artifact)

    art  = np.zeros((n, K), dtype=np.float32); art[:, art_idx] = 1.0
    seiz = np.full((n, 1), float(seizure), dtype=np.float32)
    age  = np.zeros((n, 4), dtype=np.float32); age[:, int(age_bin)] = 1.0
    mont = np.full((n, 1), float(montage_id), dtype=np.float32)

    cond = np.concatenate([art, seiz, age, mont], axis=1).astype(np.float32)
    if expect_dim is not None and cond.shape[1] != expect_dim:
        print(f"[warn] cond_dim mismatch: built {cond.shape[1]} vs --cond_dim={expect_dim}")
    return cond


def _strip_once(k, pref): return k[len(pref):] if k.startswith(pref) else k
def _clean_keys(d):
    out={}
    for k,v in d.items():
        if not torch.is_tensor(v): continue
        k=_strip_once(k,"module."); k=_strip_once(k,"net."); k=_strip_once(k,"model.")
        out[k]=v
    return out

def best_state_dict_for(net, ckpt, prefer_ema=True):
    """
    Search common locations for a UNet state_dict and pick the one with the
    largest key overlap with the target net. If prefer_ema=False, we
    downweight ema candidates slightly.
    """
    cands=[]
    if isinstance(ckpt, dict):
        # EMA variants
        ema_keys=[]
        for k in ("ema","model_ema","net_ema"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ema=ckpt[k]
                if "shadow" in ema and isinstance(ema["shadow"], dict):
                    ema_keys.append((f"{k}.shadow", ema["shadow"]))
                ema_keys.append((k, ema))
        # Non-EMA candidates
        non_ema=[]
        for k in ("state_dict","net","model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                non_ema.append((k, ckpt[k]))
        non_ema.append(("flat", ckpt))

        # Order by preference
        if prefer_ema:
            cands = ema_keys + non_ema
        else:
            cands = non_ema + ema_keys

    tgt=set(net.state_dict().keys())
    best=None; best_score=-1
    for name, cand in cands:
        cleaned=_clean_keys(cand)
        s=len(set(cleaned.keys()) & tgt)
        if s>best_score:
            best_score=s; best=cleaned
    return best, best_score


# --------------- SimpleDDIM local (eta=0, with tqdm) ---------------

class SimpleDDIM:
    """Deterministic DDIM with pseudo-CFG (zero-cond), eps-pred style, with tqdm."""
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.02, device="cpu"):
        self.T=T; self.device=device
        betas=torch.linspace(beta_min, beta_max, T, device=device)
        alphas=1.0-betas
        ac=torch.cumprod(alphas, dim=0)
        self.sqrt_ac=torch.sqrt(ac); self.sqrt_1m_ac=torch.sqrt(1.0-ac)

    def _idx(self, steps):  # evenly spaced from T-1..0
        return torch.linspace(self.T-1, 0, steps, device=self.device).long()

    @torch.no_grad()
    def sample(self, model, shape, steps=80, guidance=1.0, cond=None, cond0=None, batch=256, show_pbar=False):
        N,C,T=shape
        idx=self._idx(steps)
        outs=[]
        outer = range(0, N, batch)
        outer_iter = tqdm(outer, desc="batches", disable=not show_pbar)
        for i in outer_iter:
            bs=min(batch,N-i)
            x=torch.randn(bs,C,T,device=self.device)
            inner_iter = tqdm(range(len(idx)), desc=f"steps (batch {i//batch+1})", leave=False, disable=not show_pbar)
            for _ in inner_iter:
                ti = idx[_].expand(bs)
                a=self.sqrt_ac[ti][:,None,None]; b=self.sqrt_1m_ac[ti][:,None,None]
                if guidance>1.0 and cond0 is not None:
                    eps_c=model(x, cond[i:i+bs]); eps_u=model(x, cond0[i:i+bs])
                    eps=eps_u + guidance*(eps_c - eps_u)
                else:
                    eps=model(x, cond[i:i+bs])
                x0=(x - b*eps)/(a + 1e-8)
                if ti.min()>0:
                    tim1=(ti-1).clamp(0,self.T-1)
                    a1=self.sqrt_ac[tim1][:,None,None]; b1=self.sqrt_1m_ac[tim1][:,None,None]
                    x=a1*x0 + b1*eps
                else:
                    x=x0
            outs.append(x.detach().cpu())
        return torch.cat(outs,0)


# ---------------- main ----------------

def parse_args():
    ap=argparse.ArgumentParser()
    # model/ckpt
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--widths", type=int, nargs="+", default=[64,128,256])
    ap.add_argument("--cond_dim", type=int, default=12)
    ap.add_argument("--signal_len", type=int, default=800)

    # sampling
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--guidance", type=float, default=1.0)   # pseudo-CFG scale
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sampler", type=str, default="ddim",
                    choices=["ddim","heun2","ddim_local"])
    ap.add_argument("--tqdm", action="store_true", help="show progress bars (ddim_local only)")

    # conditioning
    ap.add_argument("--artifact", type=str, default="none")
    ap.add_argument("--seizure", type=int, default=0)
    ap.add_argument("--age_bin", type=int, default=1)
    ap.add_argument("--montage_id", type=int, default=0)
    ap.add_argument("--cond_scale", type=float, default=1.0)

    # I/O
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_npy", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


@torch.no_grad()
def main():
    args=parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=device_pick()
    print("[device]", device.type)

    C,T=8,args.signal_len

    # UNet
    net=UNet1DFiLM(c_in=C, c_hidden=tuple(args.widths), cond_dim=args.cond_dim).to(device)
    # Load ckpt (best-match; prefer EMA if --use_ema)
    ckpt=torch.load(args.ckpt, map_location=device)
    state,score=best_state_dict_for(net, ckpt, prefer_ema=bool(args.use_ema))
    if state is None or score<=0:
        raise RuntimeError("Could not find a usable UNet state_dict in checkpoint.")
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[Diffusion] Ignoring unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}")
    if missing:
        print(f"[Diffusion] Missing keys ({len(missing)}): {sorted(missing)[:8]}")
    net.eval()

    # Cond vector
    canon=None
    try:
        from models.conditioning import ARTIFACTS_CANON as _canon; canon=list(_canon)
    except Exception:
        try:
            from conditioning import ARTIFACTS_CANON as _canon2; canon=list(_canon2)
        except Exception:
            canon=None
    cond_np=build_cond_numpy(
        n=args.n, artifact=args.artifact, seizure=args.seizure,
        age_bin=args.age_bin, montage_id=args.montage_id,
        canon=canon, expect_dim=args.cond_dim
    )
    if args.cond_scale!=1.0:
        cond_np *= float(args.cond_scale)
    cond=torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)

    # Sampler
    samples=None
    try:
        if args.sampler == "ddim_local":
            # local with tqdm
            ddim=SimpleDDIM(T=1000, device=device)
            cond0=torch.zeros_like(cond)
            samples=ddim.sample(
                model=lambda x,c: net(x,c), shape=(args.n, C, T),
                steps=args.steps, guidance=float(args.guidance),
                cond=cond, cond0=cond0, batch=args.batch, show_pbar=args.tqdm
            )
        else:
            # repo sampler (no per-step tqdm hooks available)
            diff=ModelDiffusion(model=net, T=1000).to(device)
            if args.sampler=="ddim":
                samples = diff.ddim_sample(
                    n=args.n, cond=cond, steps=args.steps,
                    guidance=max(0.0, float(args.guidance)),
                    eta=0.0, batch=args.batch, shape=(C,T), device=device
                )
            else:  # heun2
                samples = diff.heun2_sample(
                    n=args.n, cond=cond, steps=max(2, args.steps//4),
                    guidance=max(0.0, float(args.guidance)),
                    batch=args.batch, shape=(C,T), device=device
                )
            samples = samples.to(device)
    except Exception as e:
        print("[warn] Repo sampler failed; using local DDIM with tqdm. Reason:", repr(e))
        ddim=SimpleDDIM(T=1000, device=device)
        cond0=torch.zeros_like(cond)
        samples=ddim.sample(
            model=lambda x,c: net(x,c), shape=(args.n, C, T),
            steps=args.steps, guidance=float(args.guidance),
            cond=cond, cond0=cond0, batch=args.batch, show_pbar=args.tqdm
        )

    x = samples.detach().cpu().float().numpy()
    if args.save_npy:
        np.save(os.path.join(args.out_dir, "samples.npy"), x.astype(np.float32))
        print(f"[save] {os.path.join(args.out_dir,'samples.npy')}  shape={x.shape}")
    print("[done] sampling complete")


if __name__ == "__main__":
    main()
