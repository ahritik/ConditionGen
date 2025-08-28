# models/diffusion.py
"""
Diffusion core with cosine noise schedule, v-prediction, SNR-weighted loss,
and DDIM / Heun2 samplers.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------
# Noise schedule (cosine)
# -----------------------
def _alpha_bar_cosine(t, s=0.008):
    # t in [0,1]
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

@dataclass
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sigmas: torch.Tensor

def make_cosine_schedule(T: int, device):
    ts = torch.linspace(0, 1, T+1, device=device)
    ab = _alpha_bar_cosine(ts)  # length T+1
    ab = torch.clamp(ab, 1e-5, 0.99999)
    betas = 1 - (ab[1:] / ab[:-1])
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sigmas = torch.sqrt(betas)
    return Schedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars, sigmas=sigmas)

# -----------------------
# v-pred helpers
# -----------------------
def x0_from_x_t_v(x_t, v, alpha_bar_t):
    # DDPM relationship (Imagen-style v-pred):
    # v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
    # solve for x0
    sqrt_ab = torch.sqrt(alpha_bar_t)
    sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
    x0 = (sqrt_ab * x_t - v) / sqrt_1mab
    return x0

def eps_from_x_t_v(x_t, v, alpha_bar_t):
    sqrt_ab = torch.sqrt(alpha_bar_t)
    sqrt_1mab = torch.sqrt(1 - alpha_bar_t)
    eps = (v + sqrt_1mab * x_t) / sqrt_ab
    return eps

# -----------------------
# EMA
# -----------------------
class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.clone().detach() for k,p in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k,p in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

# -----------------------
# Loss (SNR-weighted MSE + lambda * STFT-L1 optional)
# -----------------------
def snr_weight(alpha_bar_t):
    # Following "Improved Denoising Diffusion..." style weighting
    # weight = min(SNR, cap) where SNR = alpha_bar / (1-alpha_bar)
    snr = alpha_bar_t / (1 - alpha_bar_t + 1e-8)
    return torch.clamp(snr, 0., 5.)

def stft_l1(x, y, n_fft=128, hop_length=64):
    # Simple STFT L1 distance across channels
    # x,y: (B,C,T)
    import torch
    import torch.fft as fft
    win = torch.hann_window(n_fft, device=x.device).unsqueeze(0)  # (1,n_fft)
    def _spec(z):
        # strided windows
        B,C,T = z.shape
        unfold = z.unfold(dimension=2, size=n_fft, step=hop_length)  # (B,C,nw,n_fft)
        win_z = unfold * win.unsqueeze(0).unsqueeze(0)
        spec = torch.view_as_complex(torch.fft.rfft(win_z, dim=-1))  # (B,C,nw, n_fft//2+1)
        mag = spec.abs()
        return mag
    return ( _spec(x) - _spec(y) ).abs().mean()

class Diffusion(nn.Module):
    def __init__(self, model: nn.Module, T=1000, stft_lambda=0.1, device='cpu'):
        super().__init__()
        self.model = model
        self.T = T
        self.stft_lambda = stft_lambda
        self.register_buffer('t_sched_dummy', torch.tensor(0.))  # just to bind device
        self.schedule = None
        self._build_schedule()

    def _build_schedule(self):
        device = self.t_sched_dummy.device
        self.schedule = make_cosine_schedule(self.T, device=device)

    def _gammas_betas_per_block(self, cond_embed_module, cond_dict, num_blocks):
        # build a list of (gamma, beta) pairs, one per ResBlock (down+mid+up approx.)
        gamma, beta = cond_embed_module(cond_dict)  # (B, film_dim)
        # replicate for as many blocks as needed
        return [(gamma, beta)] * num_blocks

    def forward(self, x0, cond_dict):
        """
        Training loss for one batch.
        x0: (B,C,T) target clean signal
        cond_dict: conditioning tensors
        """
        B = x0.shape[0]
        device = x0.device
        # random t per sample
        t = torch.randint(0, self.T, (B,), device=device)
        alpha_bar_t = self.schedule.alpha_bars[t].view(B,1,1)
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1mab = torch.sqrt(1 - alpha_bar_t)

        # sample noise
        eps = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps

        # predict v
        v = self.model(x_t, self._all_block_cond(cond_dict))

        # v-target
        v_target = sqrt_ab * eps - sqrt_1mab * x0

        w = snr_weight(alpha_bar_t)
        mse = ((v - v_target)**2).mean(dim=(1,2)) * w.squeeze()
        loss_mse = mse.mean()

        if self.stft_lambda > 0:
            x0_pred = x0_from_x_t_v(x_t, v, alpha_bar_t)
            loss_stft = stft_l1(x0_pred, x0)
        else:
            loss_stft = torch.tensor(0., device=device)

        loss = loss_mse + self.stft_lambda * loss_stft
        return loss, {"loss_mse": loss_mse.detach(), "loss_stft": loss_stft.detach()}

    def _all_block_cond(self, cond_dict):
        # Model expects a list of gammas/betas for each ResBlock traversal.
        # For simplicity, produce a fixed-length list inferred from model structure.
        # We'll approximate: total blocks = len(downs_res) + 1(mid) + len(ups_res)
        # Here we set an attribute on the model after first call to cache this number.
        if not hasattr(self.model, "_approx_num_blocks"):
            # Heuristic: count ResBlocks
            n = 0
            for m in self.model.modules():
                class_name = m.__class__.__name__
                if class_name == "ResBlock":
                    n += 1
            self.model._approx_num_blocks = n + 1  # +1 for mid
        # Use the model's film_provider (ConditionEmbed) for FiLM dim inference
        gamma, beta = self.model.film_provider(cond_dict)
        gb = [(gamma, beta)] * self.model._approx_num_blocks
        return gb

    @torch.no_grad()
    def ddim_sample(self, x_T, cond_dict, steps=50, eta=0.0):
        """
        Non-ancestral DDIM sampling (eta ~ 0); predicts v and steps down.
        """
        device = x_T.device
        t_seq = torch.linspace(self.T-1, 0, steps, device=device).long()
        x = x_T
        for i, t in enumerate(t_seq):
            ab_t = self.schedule.alpha_bars[t]
            ab_prev = self.schedule.alpha_bars[max(t-1, 0)]
            x_in = x
            v = self.model(x_in, self._all_block_cond(cond_dict))
            x0 = x0_from_x_t_v(x_in, v, ab_t)
            # DDIM update
            sigma_t = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t/ab_prev))
            dir_xt = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev - sigma_t**2) * eps_from_x_t_v(x_in, v, ab_t)
            if sigma_t > 0:
                noise = torch.randn_like(x)
                x = dir_xt + sigma_t * noise
            else:
                x = dir_xt
        return x

    @torch.no_grad()
    def heun2_sample(self, x_T, cond_dict, steps=20):
        """
        Heun's 2nd order sampler in t-space (coarse but fast).
        """
        device = x_T.device
        t_seq = torch.linspace(self.T-1, 0, steps, device=device).long()
        x = x_T
        for i, t in enumerate(t_seq):
            ab_t = self.schedule.alpha_bars[t]
            v1 = self.model(x, self._all_block_cond(cond_dict))
            # Euler
            eps1 = eps_from_x_t_v(x, v1, ab_t)
            # Predict next x (coarse)
            if i+1 < len(t_seq):
                t_next = t_seq[i+1]
            else:
                t_next = torch.tensor(0, device=device)
            ab_n = self.schedule.alpha_bars[t_next]
            x_euler = torch.sqrt(ab_n) * x0_from_x_t_v(x, v1, ab_t) + torch.sqrt(1 - ab_n) * eps1
            # Corrector
            v2 = self.model(x_euler, self._all_block_cond(cond_dict))
            eps2 = eps_from_x_t_v(x_euler, v2, ab_n)
            eps_avg = 0.5*(eps1 + eps2)
            x = torch.sqrt(ab_n) * x0_from_x_t_v(x, v2, ab_t) + torch.sqrt(1 - ab_n) * eps_avg
        return x
