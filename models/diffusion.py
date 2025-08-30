# models/diffusion.py
# Small 1D VP-Diffusion with v-prediction, cosine schedule, SNR-weighted loss,
# DDIM & Heun2 samplers, and Hann-window STFT-L1 auxiliary loss.
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------- Schedules ----------------

def _cosine_alpha_bar(T: int, s: float = 0.008, eps: float = 1e-12) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal). Returns alpha_bar midpoint ratios."""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    f = f / f[0]
    abar_mid = torch.clamp(f[1:] / f[:-1], min=eps, max=1.0 - eps)
    alpha_bar = torch.cumprod(abar_mid.to(torch.float32), dim=0)
    return alpha_bar


def _gather(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Gather 1D schedule buffer x at indices t (long) -> [B,1,1]."""
    return x.index_select(0, t).view(-1, 1, 1)


# ---------------- Diffusion wrapper ----------------

class Diffusion(nn.Module):
    """
    Training:
      - v-pred objective with SNR-weighted MSE
      - optional λ * STFT-L1 (Hann window)
    Sampling:
      - DDIM and Heun2 ODE samplers
    Expects model(x_t, cond_vec) -> v_pred with shape [B,C,T].
    """
    def __init__(
        self,
        model: nn.Module,
        T: int = 1000,
        stft_win: int = 128,
        stft_hop: int = 64,
        lambda_stft: float = 0.1,
        snr_clip: float = 5.0,
        schedule: str = "cosine",
        **legacy_kwargs,   # accept legacy arg names
    ):
        super().__init__()

        # ---- back-compat mapping (so legacy train.py calls still work) ----
        if "timesteps" in legacy_kwargs:
            T = int(legacy_kwargs.pop("timesteps"))
        if "stft_lambda" in legacy_kwargs:
            lambda_stft = float(legacy_kwargs.pop("stft_lambda"))
        if "stft_cfg" in legacy_kwargs:
            # expected (win, hop, win_len); if win_len present, use as n_fft/win_length
            cfg = legacy_kwargs.pop("stft_cfg")
            try:
                stft_win = int(cfg[0])
                stft_hop = int(cfg[1])
                if len(cfg) >= 3 and int(cfg[2]) > 0:
                    stft_win = int(cfg[2])
            except Exception:
                pass
        if legacy_kwargs:
            print(f"[Diffusion] Note: ignored legacy kwargs: {list(legacy_kwargs.keys())}")

        # store final (possibly remapped) args
        self.model = model
        self.T = int(T)
        self.timesteps = self.T  # back-compat for legacy code
        self.stft_win = int(stft_win)
        self.stft_hop = int(stft_hop)
        self.lambda_stft = float(lambda_stft)
        self.snr_clip = float(snr_clip)
        self.schedule = schedule

        # ---- schedule ----
        if schedule == "cosine":
            alpha_bar = _cosine_alpha_bar(self.T)  # [T] float32
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = torch.ones(self.T, dtype=torch.float32)
        alphas[0] = alpha_bar[0]
        alphas[1:] = alpha_bar[1:] / alpha_bar[:-1]
        betas = 1.0 - alphas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        # ✅ make persistent so they’re saved/loaded with ckpts (older runs expect these)
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=True)

        sqrt_ab = torch.sqrt(alphas_cumprod)
        sqrt_1mab = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_ab, persistent=True)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_1mab, persistent=True)

        # aliases for older checkpoints
        self.register_buffer("c0", sqrt_ab.clone(), persistent=True)
        self.register_buffer("c1", sqrt_1mab.clone(), persistent=True)

        # Hann window for STFT-L1 (keep non-persistent so old ckpts don’t complain)
        self.register_buffer("stft_window", torch.hann_window(self.stft_win, periodic=True), persistent=False)

    # ----- v-parameterization helpers -----
    @staticmethod
    def _x0_from_v(x_t: torch.Tensor, v: torch.Tensor, a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return a_t * x_t - b_t * v

    @staticmethod
    def _eps_from_v(x_t: torch.Tensor, v: torch.Tensor, a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
        return b_t * x_t + a_t * v

    # ----- STFT magnitude -----
    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T] -> |STFT|: [B, C, F, L]
        """
        B, C, T = x.shape
        win = self.stft_window.to(device=x.device, dtype=x.dtype)
        X = torch.stft(
            x.reshape(B * C, T),
            n_fft=self.stft_win,
            hop_length=self.stft_hop,
            win_length=self.stft_win,
            window=win,
            center=True,
            return_complex=True,
        )
        mag = X.abs()
        return mag.view(B, C, mag.shape[-2], mag.shape[-1])

    # ----- training -----
    def forward(
        self,
        x0: torch.Tensor,            # [B,C,T]
        cond_vec: torch.Tensor,      # [B,D]
        t: Optional[torch.Tensor] = None,  # [B] long
    ):
        device = x0.device
        B, C, T = x0.shape

        if t is None:
            t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        elif t.dtype != torch.long:
            t = t.to(torch.long)

        # gather schedule on correct device/dtype
        a_t = _gather(self.sqrt_alphas_cumprod, t.to(self.sqrt_alphas_cumprod.device)).to(device=device, dtype=x0.dtype)
        b_t = _gather(self.sqrt_one_minus_alphas_cumprod, t.to(self.sqrt_one_minus_alphas_cumprod.device)).to(device=device, dtype=x0.dtype)

        eps = torch.randn_like(x0)
        x_t = a_t * x0 + b_t * eps

        # target v
        v_target = a_t * eps - b_t * x0

        # predict v
        v_pred = self.model(x_t, cond_vec)

        # SNR-weighted v-MSE
        snr_t = (a_t.squeeze(-1).squeeze(-1) ** 2) / (b_t.squeeze(-1).squeeze(-1) ** 2 + 1e-12)
        w = torch.clamp(snr_t, max=self.snr_clip)
        w = w / (w.mean() + 1e-8)
        mse = (v_pred - v_target).pow(2).mean(dim=(1, 2))
        base_loss = (w * mse).mean()

        loss = base_loss
        stft_l1_val = torch.tensor(0.0, device=device, dtype=x0.dtype)

        if self.lambda_stft > 0.0:
            # STFT-L1 on reconstructed x0 (compute in fp32 for stability)
            xr = x0.float()
            x0_pred = self._x0_from_v(x_t.float(), v_pred.float(), a_t.float(), b_t.float())
            mag_r = self._stft_mag(xr)
            mag_p = self._stft_mag(x0_pred)
            stft_l1_val = (mag_r - mag_p).abs().mean().to(x0.dtype)
            loss = loss + self.lambda_stft * stft_l1_val

        parts = {
            "base": float(base_loss.detach().cpu()),
            "stft_l1": float(stft_l1_val.detach().cpu()),
            "snr_mean": float(snr_t.mean().detach().cpu()),
        }
        return loss, parts

    # ----- DDIM sampler -----
    @torch.no_grad()
    def ddim_sample(
        self,
        n: int,
        cond: torch.Tensor,
        steps: int = 50,
        guidance: float = 0.0,
        eta: float = 0.0,
        batch: int = 256,
        shape: Tuple[int, int] = (8, 800),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        C, T = shape
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        step_idx = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)

        outs = []
        for i0 in range(0, n, batch):
            i1 = min(n, i0 + batch)
            bs = i1 - i0
            c = cond[i0:i1].to(device, dtype=dtype)
            x = torch.randn(bs, C, T, device=device, dtype=dtype)

            for j in range(steps):
                t = step_idx[j].expand(bs)
                a_t = _gather(self.sqrt_alphas_cumprod, t.to(self.sqrt_alphas_cumprod.device)).to(device=device, dtype=dtype)
                b_t = _gather(self.sqrt_one_minus_alphas_cumprod, t.to(self.sqrt_one_minus_alphas_cumprod.device)).to(device=device, dtype=dtype)

                if guidance > 0.0:
                    v_un = self.model(x, torch.zeros_like(c))
                    v_co = self.model(x, c)
                    v = v_un + guidance * (v_co - v_un)
                else:
                    v = self.model(x, c)

                eps = self._eps_from_v(x, v, a_t, b_t)
                x0_pred = self._x0_from_v(x, v, a_t, b_t)

                t_prev = torch.clamp(t - 1, min=0)
                a_prev = _gather(self.sqrt_alphas_cumprod, t_prev.to(self.sqrt_alphas_cumprod.device)).to(device=device, dtype=dtype)
                b_prev = _gather(self.sqrt_one_minus_alphas_cumprod, t_prev.to(self.sqrt_one_minus_alphas_cumprod.device)).to(device=device, dtype=dtype)

                if eta == 0.0:
                    x = a_prev * x0_pred + b_prev * eps
                else:
                    sigma_t = eta * torch.sqrt(
                        (b_prev ** 2) * (1.0 - (a_prev ** 2) / (a_t ** 2)) / (b_t ** 2 + 1e-12)
                    )
                    z = torch.randn_like(x)
                    x = a_prev * x0_pred + torch.sqrt(torch.clamp(b_prev ** 2 - sigma_t ** 2, min=0.0)) * eps + sigma_t * z

            outs.append(x0_pred.detach().cpu())

        return torch.cat(outs, dim=0)[:n]

    # ----- Heun2 sampler -----
    @torch.no_grad()
    def heun2_sample(
        self,
        n: int,
        cond: torch.Tensor,
        steps: int = 20,
        guidance: float = 0.0,
        batch: int = 256,
        shape: Tuple[int, int] = (8, 800),
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        C, T = shape
        device = device or next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        t_cont = torch.linspace(self.T - 1, 0, steps + 1, device=device, dtype=torch.float32)

        outs = []
        for i0 in range(0, n, batch):
            i1 = min(n, i0 + batch)
            bs = i1 - i0
            c = cond[i0:i1].to(device, dtype=dtype)
            x = torch.randn(bs, C, T, device=device, dtype=dtype)

            for j in range(steps):
                t  = t_cont[j].long().expand(bs)
                tn = t_cont[j + 1].long().expand(bs)

                a_t  = _gather(self.sqrt_alphas_cumprod, t.to(self.sqrt_alphas_cumprod.device)).to(device=device, dtype=dtype)
                b_t  = _gather(self.sqrt_one_minus_alphas_cumprod, t.to(self.sqrt_one_minus_alphas_cumprod.device)).to(device=device, dtype=dtype)
                a_tn = _gather(self.sqrt_alphas_cumprod, tn.to(self.sqrt_alphas_cumprod.device)).to(device=device, dtype=dtype)
                b_tn = _gather(self.sqrt_one_minus_alphas_cumprod, tn.to(self.sqrt_one_minus_alphas_cumprod.device)).to(device=device, dtype=dtype)

                if guidance > 0.0:
                    v_un = self.model(x, torch.zeros_like(c))
                    v_co = self.model(x, c)
                    v_t = v_un + guidance * (v_co - v_un)
                else:
                    v_t = self.model(x, c)

                eps_t = self._eps_from_v(x, v_t, a_t, b_t)
                x0_t  = self._x0_from_v(x, v_t, a_t, b_t)
                x_pred = a_tn * x0_t + b_tn * eps_t

                if guidance > 0.0:
                    v_un2 = self.model(x_pred, torch.zeros_like(c))
                    v_co2 = self.model(x_pred, c)
                    v_tn  = v_un2 + guidance * (v_co2 - v_un2)
                else:
                    v_tn  = self.model(x_pred, c)

                eps_tn = self._eps_from_v(x_pred, v_tn, a_tn, b_tn)
                x0_tn  = self._x0_from_v(x_pred, v_tn, a_tn, b_tn)

                eps_avg = 0.5 * (eps_t + eps_tn)
                x0_avg  = 0.5 * (x0_t + x0_tn)
                x = a_tn * x0_avg + b_tn * eps_avg

            outs.append(x0_tn.detach().cpu())

        return torch.cat(outs, dim=0)[:n]

    # ----- lenient loader for back-compat -----
    def load_state_dict(self, state_dict, strict: bool = True):
        own = super().state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in own}
        unexpected = [k for k in state_dict.keys() if k not in own]
        missing = [k for k in own.keys() if k not in filtered]
        if unexpected:
            print(f"[Diffusion] Ignoring unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
        if missing and strict:
            print(f"[Diffusion] Missing keys (ok): {missing[:8]}{' ...' if len(missing)>8 else ''}")
        return super().load_state_dict(filtered, strict=False)


# ---------------- EMA (for sampling checkpoints) ----------------

class EMA(nn.Module):
    """
    Exponential Moving Average of model params.

    Back-compat:
      - accepts `beta=` (alias of `decay`)
      - load_state_dict() tolerates older keys like 'params'
    """
    def __init__(self, model: nn.Module, decay: float | None = None, beta: float | None = None, **kwargs):
        super().__init__()
        if decay is None and beta is not None:
            decay = beta
        if decay is None:
            decay = 0.999
        self.decay = float(decay)
        if kwargs:
            print(f"[EMA] Ignored extra kwargs: {list(kwargs.keys())}")

        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long), persistent=False)
        # shadow params (only requires_grad ones)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    def to(self, *args, **kwargs):
        device = kwargs.get("device", None)
        dtype  = kwargs.get("dtype", None)
        if device is not None or dtype is not None:
            for k in list(self.shadow.keys()):
                self.shadow[k] = self.shadow[k].to(
                    device=device or self.shadow[k].device,
                    dtype=dtype  or self.shadow[k].dtype
                )
        return super().to(*args, **kwargs)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": int(self.num_updates.item()),
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        # tolerate older formats
        if "shadow" not in state and "params" in state:
            state = {**state, "shadow": state["params"]}
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = torch.tensor(state.get("num_updates", 0), dtype=torch.long)
        sh = state.get("shadow", {})
        # only load keys that exist in current model
        for k, v in sh.items():
            if k in self.shadow:
                self.shadow[k] = v.clone()


# Back-compat export so older code can `from models.diffusion import Diffusion1D, EMA`
Diffusion1D = Diffusion
__all__ = ["Diffusion", "Diffusion1D", "EMA"]
