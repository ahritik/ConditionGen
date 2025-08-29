import torch, torch.nn as nn
from einops import rearrange

class FiLM(nn.Module):
    def __init__(self, hidden, cond_dim):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim, hidden*2)
        )
    def forward(self, x, cond):
        gb = self.to_gamma_beta(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = rearrange(gamma, 'b c -> b c 1')
        beta  = rearrange(beta, 'b c -> b c 1')
        return x * (1 + gamma) + beta

class DWConv1d(nn.Module):
    def __init__(self, c, k=5, d=1):
        super().__init__()
        self.pad = nn.ReplicationPad1d(((k-1)*d)//2)
        self.dw = nn.Conv1d(c, c, k, groups=c, dilation=d, padding=0)
        self.pw = nn.Conv1d(c, c, 1)
    def forward(self, x):  # [B,C,T]
        return self.pw(self.dw(self.pad(x)))

class ResBlock(nn.Module):
    def __init__(self, c, cond_dim, d=1):
        super().__init__()
        self.conv1 = DWConv1d(c, k=5, d=d)
        self.act1 = nn.SiLU()
        self.film1 = FiLM(c, cond_dim)
        self.conv2 = DWConv1d(c, k=5, d=d)
        self.act2 = nn.SiLU()
        self.film2 = FiLM(c, cond_dim)
        self.skip = nn.Identity()
    def forward(self, x, cond):
        h = self.conv1(x); h = self.act1(h); h = self.film1(h, cond)
        h = self.conv2(h); h = self.act2(h); h = self.film2(h, cond)
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ci, co, cond_dim):
        super().__init__()
        self.proj = nn.Conv1d(ci, co, 3, stride=2, padding=1)
        self.rb1 = ResBlock(co, cond_dim, d=1)
        self.rb2 = ResBlock(co, cond_dim, d=2)
    def forward(self, x, cond):
        x = self.proj(x)
        x = self.rb1(x, cond)
        x = self.rb2(x, cond)
        return x

class Up(nn.Module):
    def __init__(self, ci, co, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(ci, co, 4, stride=2, padding=1)
        self.rb1 = ResBlock(co, cond_dim, d=1)
        self.rb2 = ResBlock(co, cond_dim, d=2)
    def forward(self, x, skip, cond):
        x = self.up(x)
        x = x + skip
        x = self.rb1(x, cond)
        x = self.rb2(x, cond)
        return x

class UNet1DFiLM(nn.Module):
    def __init__(self, c_in=8, c_hidden=(64,128,256), cond_dim=128):
        super().__init__()
        c1, c2, c3 = c_hidden
        self.inp = nn.Conv1d(c_in, c1, 3, padding=1)
        self.d1 = Down(c1, c2, cond_dim)
        self.d2 = Down(c2, c3, cond_dim)
        self.mid1 = ResBlock(c3, cond_dim, d=2)
        self.mid2 = ResBlock(c3, cond_dim, d=4)
        self.u1 = Up(c3, c2, cond_dim)
        self.u2 = Up(c2, c1, cond_dim)
        self.out = nn.Conv1d(c1, c_in, 3, padding=1)

    def forward(self, x, cond_vec):
        # cond_vec: [B, cond_dim] already projected before if needed
        x1 = self.inp(x)
        s1 = self.d1(x1, cond_vec)
        s2 = self.d2(s1, cond_vec)
        m  = self.mid1(s2, cond_vec)
        m  = self.mid2(m, cond_vec)
        u1 = self.u1(m, s2, cond_vec)
        u2 = self.u2(u1, s1, cond_vec)
        out = self.out(u2)
        return out
