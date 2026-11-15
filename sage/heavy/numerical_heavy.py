"""Numerical large model + learned verifier: the non-LLM heavy path for the 'numerical_heavy' variant.

Patch-embedding Transformer encoder (frozen-free, trained on the fit partition only). Its pooled latent
feeds a linear sigmoid verifier trained on realised error labels (CoLLM eqs. 5, 12-14). Same routing
(gate -> heavy -> confidence-gap -> average) as SAGE, but the heavy component is purely numerical.
"""
import torch
from torch import Tensor, nn


class NumericalLarge(nn.Module):
    def __init__(self, window: int = 50, features: int = 14, d_model: int = 128, layers: int = 4,
                 heads: int = 4, patch: int = 5, dropout: float = 0.1):
        super().__init__()
        if window % patch:
            raise ValueError("window must be divisible by patch")
        self.window, self.features, self.patch, self.d_model = window, features, patch, d_model
        self.embed = nn.Linear(patch * features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window // patch, d_model))
        block = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b = x.shape[0]
        tokens = self.embed(x.reshape(b, self.window // self.patch, self.patch * self.features)) + self.pos
        latent = self.norm(self.encoder(tokens)).mean(dim=1)
        return self.head(latent) * self.target_std + self.target_mean, latent


class NumericalVerifier(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.projection = nn.Linear(d_model, 1)

    def forward(self, latent: Tensor) -> Tensor:
        return torch.sigmoid(self.projection(latent))
