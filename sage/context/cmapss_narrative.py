"""SAGE sensor-to-text adaptation, not an equation or claim from either paper."""
import numpy as np
from sage.config import Config

SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

def narrate(window: np.ndarray, config: Config) -> list[str]:
    """Causal within each observed window. No RUL, failure date, or test statistics.

    z scores flag relative deviations, NOT validated physical failure thresholds.
    """
    if window.ndim != 2 or window.shape[1] != 14:
        raise ValueError("CMAPSS requires the retained 14 sensors")
    result = []
    for t, row in enumerate(window):
        if t < 2:
            result.append("Insufficient prior observations for a rolling deviation estimate.")
            continue
        prior = window[max(0, t-config.narrative_window):t]
        scale = prior.std(axis=0)
        safe = np.where(scale > 1e-8, scale, 1.)
        z = (row - prior.mean(axis=0)) / safe
        flagged = np.flatnonzero((scale > 1e-8) & (np.abs(z) >= config.narrative_z))
        ranked = sorted(flagged, key=lambda i: -abs(z[i]))[:3]
        sentences = [f"Sensor {SENSORS[i]} is {abs(z[i]):.2f} rolling standard deviations "
                     f"{'above' if z[i] > 0 else 'below'} its prior mean; latest change {row[i]-window[t-1,i]:+.3f}."
                     for i in ranked]
        result.append(" ".join(sentences) or "No large relative sensor deviations in this window.")
    return result
