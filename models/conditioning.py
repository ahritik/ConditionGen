import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_channels:int, cond_dim:int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, in_channels)
        self.to_beta  = nn.Linear(cond_dim, in_channels)
    def forward(self, x, cond):
        # x: (B, C, T), cond: (B, D)
        gamma = self.to_gamma(cond).unsqueeze(-1)
        beta  = self.to_beta(cond).unsqueeze(-1)
        return x * (1 + gamma) + beta

class CondEmbedding(nn.Module):
    def __init__(self, artifact_k:int, intensity_dim:int=1, d_model:int=128, p_drop:float=0.1):
        super().__init__()
        self.artifact_emb = nn.Embedding(artifact_k, d_model)
        self.intensity_mlp = nn.Sequential(
            nn.Linear(intensity_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(p_drop)
    def forward(self, artifact_idx, intensity_scalar):
        # artifact_idx: (B,), intensity_scalar: (B,1)
        a = self.artifact_emb(artifact_idx)
        s = self.intensity_mlp(intensity_scalar)
        return self.dropout(a + s)  # (B, d_model)
