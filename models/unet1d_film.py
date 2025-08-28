# models/unet1d_film.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size // 2) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x, gamma, beta):
        # x: (B,C,T), gamma/beta: (B,C)
        # reshape gamma/beta to (B,C,1)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(self, ch, cond_dim=None, dilation=1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(ch, ch, kernel_size=3, dilation=dilation)
        self.conv2 = DepthwiseSeparableConv1d(ch, ch, kernel_size=3, dilation=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()
        self.film = FiLM(ch)

    def forward(self, x, gamma, beta):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.film(h, gamma, beta)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet1DFiLM(nn.Module):
    """
    A small 1D UNet with FiLM modulation per block.
    Input: (B, C, T) -> predicts v (velocity) for v-pred diffusion.
    """
    def __init__(self, in_channels=8, base=64, widths=(64,128,256), num_res_blocks=2, film_provider=None):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.widths = widths
        self.num_res_blocks = num_res_blocks
        self.film_provider = film_provider

        self.in_conv = nn.Conv1d(in_channels, widths[0], kernel_size=3, padding=1)

        # Down path
        downs = []
        ch = widths[0]
        for w in widths:
            for _ in range(num_res_blocks):
                downs.append(ResBlock(w))
            if w != widths[-1]:
                downs.append(Down(w, w*2))
        self.downs = nn.ModuleList(downs)

        # Middle
        mid_ch = widths[-1]*2 if len(widths) > 1 else widths[-1]
        self.mid_in = nn.Conv1d(widths[-1], mid_ch, kernel_size=3, padding=1)
        self.mid_rb = ResBlock(mid_ch, dilation=2)
        self.mid_out = nn.Conv1d(mid_ch, widths[-1], kernel_size=3, padding=1)

        # Up path
        ups = []
        up_ws = list(widths)[::-1]
        ch = widths[-1]
        for i, w in enumerate(up_ws):
            for _ in range(num_res_blocks):
                ups.append(ResBlock(ch + w))  # skip concat
            if i != len(up_ws) - 1:
                ups.append(Up(ch + w, ch // 2))
                ch = ch // 2
        self.ups = nn.ModuleList(ups)

        # Out
        self.out_norm = nn.GroupNorm(8, ch + widths[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(ch + widths[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x, gammas_betas):
        """
        gammas_betas: list of (gamma, beta) tensors for each ResBlock encountered (down + mid + up + out pre).
        We will pop from the list in order.
        """
        gb_iter = iter(gammas_betas)

        skips = []
        h = self.in_conv(x)

        # Down path
        down_feats = []
        idx = 0
        for mod in self.downs:
            if isinstance(mod, ResBlock):
                gamma, beta = next(gb_iter)
                h = mod(h, gamma, beta)
                down_feats.append(h)
            else:
                skips.append(h)
                h = mod(h)

        # Middle
        h = self.mid_in(h)
        gamma, beta = next(gb_iter)
        h = self.mid_rb(h, gamma, beta)
        h = self.mid_out(h)

        # Up path
        up_skips = list(reversed(down_feats[:len(self.ups)]))  # approximate matching
        k = 0
        for mod in self.ups:
            if isinstance(mod, ResBlock):
                skip = up_skips[k] if k < len(up_skips) else None
                if skip is not None:
                    h = torch.cat([h, skip], dim=1)
                gamma, beta = next(gb_iter)
                h = mod(h, gamma, beta)
                k += 1
            else:
                h = mod(h)

        # Final
        if len(skips) > 0:
            h = torch.cat([h, skips[0]], dim=1)  # fuse earliest skip
        h = self.out_norm(h)
        h = self.out_act(h)
        v = self.out_conv(h)
        return v
