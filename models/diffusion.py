import math, torch, torch.nn as nn, torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

class EMA:
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def update(self, model):
        i = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            self.shadow[i].mul_(self.beta).add_(p, alpha=1-self.beta)
            i += 1
    def copy_to(self, model):
        i = 0
        for p in model.parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.shadow[i])
            i += 1

def stft_l1(x, y, n_fft=256, hop_length=128, win_length=256):
    # Compute L1 distance between magnitude STFTs, averaged over channels
    # x,y: [B,C,T]
    B,C,T = x.shape
    loss = 0.0
    for c in range(C):
        X = torch.stft(x[:,c,:], n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        Y = torch.stft(y[:,c,:], n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
        loss = loss + (X.abs() - Y.abs()).abs().mean()
    return loss / C

class Diffusion1D(nn.Module):
    """
    v-prediction objective with cosine schedule. Supports DDIM sampling.
    """
    def __init__(self, model, timesteps=1000, stft_lambda=0.1, stft_cfg=(128,64,128)):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps).float()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        self.register_buffer('c0', torch.sqrt(alphas_cumprod))
        self.register_buffer('c1', torch.sqrt(1 - alphas_cumprod))

        self.stft_lambda = stft_lambda
        self.stft_cfg = stft_cfg  # (n_fft, hop, win)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_cum = self.sqrt_alphas_cumprod[t].view(-1,1,1)
        sqrt_om  = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
        return sqrt_cum * x0 + sqrt_om * noise

    def forward(self, x0, cond_vec, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        # v-pred uses v = alpha_t * noise - sqrt(1-alpha_t) * x0
        at = self.sqrt_alphas_cumprod[t].view(-1,1,1)
        omt = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
        v_target = at * noise - omt * x0
        v_pred = self.model(xt, cond_vec)
        mse = F.mse_loss(v_pred, v_target)
        if self.stft_lambda > 0:
            # reconstruct x0_pred from v_pred and xt
            x0_pred = at * xt - omt * v_pred
            s_cfg = self.stft_cfg
            l_spec = stft_l1(x0_pred, x0, n_fft=s_cfg[0], hop_length=s_cfg[1], win_length=s_cfg[2])
            return mse + self.stft_lambda * l_spec, {"mse":mse.item(), "stft": l_spec.item()}
        return mse, {"mse":mse.item()}

    @torch.no_grad()
    def ddim_sample(self, shape, cond_vec, steps=50, eta=0.0, guidance_scale=0.0, cond_null=None, device="cpu"):
        """
        guidance_scale: classifier-free guidance; if >0, we require cond_null vector to mix.
        """
        b = shape[0]
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.timesteps-1, 0, steps, device=device, dtype=torch.long)
        for i in range(steps):
            t = ts[i].long().expand(b)
            at = self.sqrt_alphas_cumprod[t].view(-1,1,1)
            omt = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1)
            if guidance_scale > 0 and cond_null is not None:
                v_cond = self.model(x, cond_vec)
                v_null = self.model(x, cond_null)
                v = v_null + guidance_scale * (v_cond - v_null)
            else:
                v = self.model(x, cond_vec)
            x0_pred = at * x - omt * v
            if i == steps-1:
                x = x0_pred
            else:
                t_next = ts[i+1].long().expand(b)
                at_next = self.sqrt_alphas_cumprod[t_next].view(-1,1,1)
                omt_next = self.sqrt_one_minus_alphas_cumprod[t_next].view(-1,1,1)
                # DDIM deterministic update (eta=0)
                x = at_next * x0_pred + omt_next * v
        return x
