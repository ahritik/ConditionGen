import torch
import torch.nn as nn
import math
from .conditioning import FiLM

def sinusoidal_time_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, device=device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.film = FiLM(out_ch, cond_dim)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()

    def forward(self, x, t_emb, cond):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.film(h, cond)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))

class UNet1D(nn.Module):
    def __init__(self, channels:int, widths=(64,128,256), resblocks=2, time_dim=256, cond_dim=128):
        super().__init__()
        w1,w2,w3 = widths
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.in_conv = nn.Conv1d(channels, w1, 3, padding=1)

        def make_stage(inc,outc):
            blocks = []
            blocks.append(ResBlock(inc, outc, time_dim, cond_dim))
            for _ in range(resblocks-1):
                blocks.append(ResBlock(outc, outc, time_dim, cond_dim))
            return nn.ModuleList(blocks)

        self.down1 = make_stage(w1,w1)
        self.down2 = make_stage(w1,w2)
        self.down3 = make_stage(w2,w3)
        self.pool = nn.AvgPool1d(2)

        self.mid1 = ResBlock(w3,w3,time_dim,cond_dim)
        self.mid2 = ResBlock(w3,w3,time_dim,cond_dim)

        self.up3 = make_stage(w3+w2,w2)
        self.up2 = make_stage(w2+w1,w1)
        self.up1 = make_stage(w1+w1,w1)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.out = nn.Conv1d(w1, channels, 1)

    def forward(self, x, t, cond):
        # x:(B,C,T), t:(B,), cond:(B,D)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        # cond already prepared (B,cond_dim)
        h = self.in_conv(x)
        # down
        d1 = h
        for blk in self.down1:
            d1 = blk(d1, t_emb, cond)
        p1 = self.pool(d1)

        d2 = p1
        for blk in self.down2:
            d2 = blk(d2, t_emb, cond)
        p2 = self.pool(d2)

        d3 = p2
        for blk in self.down3:
            d3 = blk(d3, t_emb, cond)

        m = self.mid1(d3, t_emb, cond)
        m = self.mid2(m, t_emb, cond)

        u3 = self.upsample(m)
        u3 = torch.cat([u3, d2], dim=1)
        for blk in self.up3:
            u3 = blk(u3, t_emb, cond)

        u2 = self.upsample(u3)
        u2 = torch.cat([u2, d1], dim=1)
        for blk in self.up2:
            u2 = blk(u2, t_emb, cond)

        u1 = self.upsample(u2)
        u1 = torch.cat([u1, self.in_conv(x)], dim=1)
        for blk in self.up1:
            u1 = blk(u1, t_emb, cond)

        return self.out(u1)
