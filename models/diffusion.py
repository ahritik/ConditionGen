import torch, math
import torch.nn as nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, schedule="cosine"):
        super().__init__()
        self.model = model
        self.T = timesteps
        if schedule=="cosine":
            self.register_buffer("alphas_cumprod", self.cosine_schedule(timesteps))
        else:
            self.register_buffer("alphas_cumprod", torch.linspace(1.0, 0.0001, timesteps))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    @staticmethod
    def cosine_schedule(T, s=0.008):
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        return alphas_cumprod[1:]

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1,1,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
        return sqrt_a * x0 + sqrt_om * noise

    def p_losses(self, x0, t, cond_emb, cfg=None, stft_loss=None, stft_lambda=0.1):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = self.model(xt, t, cond_emb)
        loss_mse = F.mse_loss(eps_pred, noise)
        if stft_loss is not None:
            with torch.no_grad():
                xhat = self.predict_x0(xt, eps_pred, t)
            loss_spec = stft_loss(x0, xhat)
            return loss_mse + stft_lambda * loss_spec, {"mse": loss_mse.item(), "stft": loss_spec.item()}
        return loss_mse, {"mse": loss_mse.item()}

    def predict_x0(self, xt, eps, t):
        a = self.sqrt_alphas_cumprod[t].view(-1,1,1)
        om = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
        return (xt - om * eps) / (a + 1e-8)

    @torch.no_grad()
    def ddim_sample(self, shape, cond_emb, steps=50, eta=0.0, cfg_scale=1.5, cond_fn=None):
        B,C,T = shape
        device = next(self.model.parameters()).device
        x = torch.randn(B,C,T, device=device)
        ts = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=device)
        for i, t in enumerate(ts):
            t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            if cond_fn is None:
                eps = self.model(x, t_batch, cond_emb)
            else:
                eps = cond_fn(self.model, x, t_batch, cond_emb, cfg_scale)
            a_t = self.sqrt_alphas_cumprod[t_batch].view(-1,1,1)
            om_t = self.sqrt_one_minus_alphas_cumprod[t_batch].view(-1,1,1)
            x0 = (x - om_t * eps) / (a_t + 1e-8)
            if i < steps-1:
                # deterministic DDIM
                x = a_t * x0 + om_t * eps
            else:
                x = x0
        return x
