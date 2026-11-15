"""All hyperparameters; weather/stock defaults are provisional, not paper presets."""
from dataclasses import dataclass

PAPER_THRESHOLDS = {
    "FD001": {"A": (0.30, 0.10), "B": (0.40, 0.10), "C": (0.60, 0.05)},
    "FD003": {"A": (0.15, 0.10), "B": (0.40, 0.10), "C": (0.60, 0.05)},
}
ALPHA_DEFAULTS = {"FD001": 20.0, "FD003": 20.0, "weather": 5.0, "stock": 10.0, "ett": 5.0}

@dataclass(frozen=True)
class Config:
    track: str = "FD001"
    window: int = 50
    features: int = 14
    horizon: int = 1
    hidden: int = 32
    tau1: float = 0.60
    tau2: float = 0.05
    alpha: float = 20.0
    fusion: str = "average"
    encoder: str = "hash-demo"
    prompt_version: str = "v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str | None = None
    embedding_dim: int = 128
    seed: int = 42
    learning_rate: float = 0.001
    max_rul: float = 125.0
    narrative_window: int = 10
    narrative_z: float = 2.0
    max_tokens: int = 8192
    timeout: float = 120.0
    retries: int = 2

    def __post_init__(self) -> None:
        if self.track not in ALPHA_DEFAULTS:
            raise ValueError("Unknown track")
        if min(self.window, self.features, self.horizon, self.hidden, self.embedding_dim) < 1:
            raise ValueError("Dimensions must be positive")
        if self.alpha <= 0 or not 0 <= self.tau1 <= 1 or not -1 <= self.tau2 <= 1:
            raise ValueError("Invalid confidence settings")
        if self.fusion not in {"average", "confidence_weighted"}:
            raise ValueError("Unknown fusion strategy")
        if self.encoder not in {"hash-demo", "sentence-transformer"}:
            raise ValueError("Unknown encoder")
        if self.prompt_version not in {"v1", "v2"}:
            raise ValueError("Unknown prompt version")

    @classmethod
    def for_track(cls, track: str, **kwargs) -> "Config":
        tau1, tau2 = PAPER_THRESHOLDS.get(track, {"C": (.6, .05)})["C"]
        defaults = dict(track=track, tau1=tau1, tau2=tau2, alpha=ALPHA_DEFAULTS[track])
        defaults.update(kwargs)
        return cls(**defaults)
