import torch, torch.nn as nn
from utils.constants import ARTIFACT_SET

class ConditionEmbed(nn.Module):
    """
    Embeds conditioning signal:
      - artifact onehot (len(ARTIFACT_SET))
      - seizure flag (1)
      - age bin onehot (4)
      - montage id (scalar) -> embedding
      - intensity (scalar) -> optional scale
    Returns a latent vector fed via FiLM into each UNet block.
    """
    def __init__(self, d_model=128, n_montage=8, use_intensity=True):
        super().__init__()
        self.use_intensity = use_intensity
        in_dim = len(ARTIFACT_SET) + 1 + 4 + 1  # montage id is separate embedding
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.montage_embed = nn.Embedding(n_montage, d_model)
        if use_intensity:
            self.intensity_proj = nn.Linear(1, d_model)

    def forward(self, artifact_onehot, seizure, age_onehot, montage_id, intensity=None):
        base = torch.cat([artifact_onehot, seizure, age_onehot, montage_id], dim=-1)
        h = self.mlp(base)
        h = h + self.montage_embed(montage_id.squeeze(-1).long())
        if self.use_intensity and intensity is not None:
            h = h + self.intensity_proj(intensity)
        return h  # [B, d_model]
