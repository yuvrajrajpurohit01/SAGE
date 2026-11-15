from dataclasses import dataclass
import math
import numpy as np
from sage.config import Config

@dataclass
class DecisionOutputs:
    delta: float
    accept_heavy: bool
    prediction: list[float]
    source: str


def decide(y_s: list[float], y_l: list[float], q_s: float, q_l: float, config: Config) -> DecisionOutputs:
    if not all(math.isfinite(q) and 0 <= q <= 1 for q in (q_s, q_l)):
        raise ValueError("Confidences must be finite in [0,1]")
    s, l = np.asarray(y_s, float), np.asarray(y_l, float)
    if s.shape != l.shape or s.shape != (config.horizon,) or not np.isfinite([s, l]).all():
        raise ValueError("Invalid predictions")
    delta = q_s - q_l
    accept = delta <= config.tau2  # CoLLM eq. 6, including equality
    if accept:
        return DecisionOutputs(delta, True, l.tolist(), "heavy")
    if config.fusion == "average":
        result = (s + l) / 2  # CoLLM eq. 7 and its stated baseline instantiation
    else:
        weight = q_s / (q_s + q_l) if q_s + q_l > 0 else 0.5
        result = weight * s + (1 - weight) * l  # SAGE experimental fusion, not CoLLM eq. 7's baseline
    return DecisionOutputs(delta, False, result.tolist(), "fusion")
