"""Learned confidence of the agentic path; never uses self-reported LLM confidence."""
import hashlib
import re
import numpy as np
import torch
from torch import Tensor, nn
from sage.config import Config
from sage.heavy.nexus_pipeline import HeavyPathOutputs

class TraceEncoder:
    """Concatenate macro embedding, pooled micro embeddings, synthesis embedding.

    Frozen sentence encoder for experiments. Chunk text to limit truncation;
    hash-demo is only a deterministic lexical test double, not semantic evidence.
    No Qs, ground truth or error enters this representation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        if config.encoder == "sentence-transformer":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.embedding_model, revision=config.embedding_revision)
            if self.model.get_sentence_embedding_dimension() != config.embedding_dim:
                raise ValueError("embedding_dim must match the configured encoder")
            self.model.eval()

    def _embed(self, text: str) -> np.ndarray:
        if self.model is not None:
            ids = self.model.tokenizer.encode(text, add_special_tokens=False)
            size = max(1, self.model.max_seq_length - 2)
            chunks = [self.model.tokenizer.decode(ids[i:i+size]) for i in range(0, len(ids), size)] or [""]
            return np.asarray(self.model.encode(chunks, normalize_embeddings=True)).mean(axis=0)
        vector = np.zeros(self.config.embedding_dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            h = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(h[:4], "big") % len(vector)] += 1 if h[4] % 2 else -1
        return vector / max(float(np.linalg.norm(vector)), 1.0)

    def __call__(self, heavy: HeavyPathOutputs) -> Tensor:
        parts = [self._embed(heavy.macro.reasoning),
                 np.mean([self._embed(t) for t in heavy.micro.step_reasoning], axis=0),
                 self._embed(heavy.synthesis.reasoning)]
        return torch.tensor(np.concatenate(parts), dtype=torch.float32).unsqueeze(0)

class SelfReflectionAgent(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.projection = nn.Linear(3 * config.embedding_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.projection(features.flatten(start_dim=1)))  # CoLLM eqs. 5,12

    @staticmethod
    def target(prediction: Tensor, truth: Tensor, alpha: float) -> Tensor:
        if prediction.shape != truth.shape or prediction.ndim != 2 or alpha <= 0:
            raise ValueError("Matching [batch,horizon] tensors and positive alpha required")
        label = 1 - torch.tanh(torch.abs(prediction.detach() - truth.detach()) / alpha)  # CoLLM eq. 13
        return label.mean(dim=1, keepdim=True)  # SAGE multi-horizon extension

    @staticmethod
    def loss(confidence: Tensor, target: Tensor) -> Tensor:
        if confidence.shape != target.shape or confidence.ndim != 2 or confidence.shape[1] != 1:
            raise ValueError("One confidence scalar per sample required")
        return ((confidence - target) ** 2).sum() / confidence.shape[0]  # CoLLM eq. 14
