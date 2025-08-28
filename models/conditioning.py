# models/conditioning.py
import torch
import torch.nn as nn
from utils.constants import ARTIFACT_SET, AGE_BINS, MONTAGE_IDS

class ConditionEmbed(nn.Module):
    """
    Embed the conditioning vector and provide FiLM (scale, shift) parameters per UNet block.

    Condition vector includes:
      - artifact onehot (7)
      - artifact intensity (1)    [0..1] optional
      - seizure flag (1)          [0/1]
      - age bin onehot (4)
      - montage id embedding (learned)
    """
    def __init__(self, film_dim: int, montage_emb_dim: int = 16):
        super().__init__()
        self.num_artifacts = len(ARTIFACT_SET)
        self.num_age_bins = len(AGE_BINS)
        self.montage_emb = nn.Embedding(len(MONTAGE_IDS), montage_emb_dim)

        # Simple MLP to produce FiLM params from concatenated features
        in_dim = self.num_artifacts + 1 + 1 + self.num_age_bins + montage_emb_dim  # art onehot + intensity + seizure + age onehot + montage emb
        hidden = max(64, film_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * film_dim)  # produce [gamma, beta]
        )

    def forward(self, cond_dict):
        """
        cond_dict keys (torch tensors on same device):
          - artifact (B,) int indices into ARTIFACT_SET
          - intensity (B,1) float 0..1
          - seizure (B,1) float 0/1
          - age_bin (B,) int [0..3]
          - montage_id (B,) int
        Returns gamma, beta shaped (B, film_dim)
        """
        B = cond_dict["artifact"].shape[0]
        device = cond_dict["artifact"].device

        art_onehot = torch.zeros(B, self.num_artifacts, device=device)
        art_onehot.scatter_(1, cond_dict["artifact"].long().unsqueeze(1), 1.0)

        age_onehot = torch.zeros(B, self.num_age_bins, device=device)
        age_onehot.scatter_(1, cond_dict["age_bin"].long().unsqueeze(1), 1.0)

        montage_e = self.montage_emb(cond_dict["montage_id"].long())  # (B, Dm)

        feats = torch.cat([
            art_onehot,
            cond_dict["intensity"].float(),
            cond_dict["seizure"].float(),
            age_onehot,
            montage_e
        ], dim=1)

        gamma_beta = self.net(feats)
        film_dim = gamma_beta.shape[1] // 2
        gamma, beta = gamma_beta[:, :film_dim], gamma_beta[:, film_dim:]
        return gamma, beta
