from dataclasses import dataclass
import torch
from torch import Tensor, nn
from sage.config import Config

@dataclass
class SmallModelOutputs:
    prediction: Tensor
    features: Tensor

class SmallModel(nn.Module):
    """Conv1D + GRU; outputs targets in original units, features [B,T,H]."""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(config.features, config.hidden, 3, padding=1)
        self.gru = nn.GRU(config.hidden, config.hidden, batch_first=True)
        self.head = nn.Linear(config.hidden, config.horizon)
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def forward(self, x: Tensor) -> SmallModelOutputs:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.config.window, self.config.features):
            raise ValueError("Expected [batch, window, features]")
        features, _ = self.gru(torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2))
        y = self.head(features[:, -1]) * self.target_std + self.target_mean  # CoLLM eq. 1
        return SmallModelOutputs(y, features)
