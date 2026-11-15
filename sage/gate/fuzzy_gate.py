from dataclasses import dataclass
import torch
from torch import Tensor, nn
from sage.config import Config

@dataclass
class GateOutputs:
    confidence: Tensor
    accept_small: Tensor
    membership: Tensor


def confidence_target(prediction: Tensor, truth: Tensor, alpha: float) -> Tensor:
    """Training only. Scalar exact; multi-horizon = mean of elementwise labels."""
    if alpha <= 0 or prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("Matching [batch,horizon] tensors and positive alpha required")
    labels = 1 - torch.tanh(torch.abs(prediction.detach() - truth.detach()) / alpha)  # CoLLM eq. 10
    return labels.mean(dim=1, keepdim=True)  # SAGE extension for multi-horizon outputs

class FuzzyDecisionAgent(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.mu = nn.Parameter(torch.zeros(config.hidden))
        self.log_variance = nn.Parameter(torch.zeros(config.hidden))
        self.head = nn.Linear(config.window * config.hidden, 1)

    def forward(self, features: Tensor) -> GateOutputs:
        variance = self.log_variance.exp()  # Positive reparameterization of learnable sigma^2
        membership = torch.exp(-((features - self.mu) ** 2) / variance)  # CoLLM eq. 8
        matrix = membership  # CoLLM eq. 9: [B,T,d], all dimensions and timesteps retained
        confidence = torch.sigmoid(self.head(matrix.flatten(start_dim=1)))  # CoLLM eqs. 2,11
        accept = confidence >= self.config.tau1  # CoLLM eq. 3, including equality
        return GateOutputs(confidence, accept, matrix)
