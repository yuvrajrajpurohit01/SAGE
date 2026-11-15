"""Forecast an unseen, already-observed history without target labels."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sage.config import Config
from sage.context.cmapss_narrative import narrate
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent
from sage.heavy.nexus_pipeline import ForecastInputs, NexusPipeline, task_instruction
from sage.pipeline import SAGEPipeline
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder


def predict_file(checkpoint_path: str | Path, history_path: str | Path, heavy: NexusPipeline) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not checkpoint.get("trained") or checkpoint["backend"] != heavy.backend.identity:
        raise ValueError("A trained checkpoint matching the heavy backend is required")
    config = Config(**checkpoint["config"])
    obj = json.loads(Path(history_path).read_text(encoding="utf-8"))
    allowed = {"values", "timestamps", "context", "available_at", "feature_names"}
    if set(obj)-allowed:
        raise ValueError("History JSON contains unknown fields; labels and future forecasts are not inference inputs")
    if obj["feature_names"] != checkpoint["feature_names"]:
        raise ValueError("Feature names/order must match training")
    raw = np.asarray(obj["values"], dtype=np.float32)
    if raw.shape != (config.window, config.features) or not np.isfinite(raw).all():
        raise ValueError("Expected one complete finite history window")
    context = obj.get("context", [""]*config.window)
    if config.track.startswith("FD"):
        context = narrate(raw, config) if checkpoint["narrative"] else [""]*config.window
    else:
        timestamps = pd.to_datetime(obj["timestamps"], utc=True, errors="raise")
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError("Observation timestamps must be strictly increasing")
        if any(context):
            available = pd.to_datetime(obj.get("available_at", []), utc=True, errors="coerce")
            if len(available) != len(context):
                raise ValueError("Text requires aligned available_at timestamps")
            if any(text and (pd.isna(a) or a > t) for text, a, t in zip(context, available, timestamps)):
                raise ValueError("Undated or future text is not allowed")
    inputs = ForecastInputs(raw.tolist(), obj["timestamps"], context, obj["feature_names"],
        task_instruction(config.prompt_version, config.max_rul) if checkpoint["target_index"] is None
        else f"Predict future {obj['feature_names'][checkpoint['target_index']]} in original units.",
        config.horizon, checkpoint["target_index"])
    small, gate, reflection = SmallModel(config), FuzzyDecisionAgent(config), SelfReflectionAgent(config)
    for name, model in [("small", small), ("gate", gate), ("reflection", reflection)]:
        model.load_state_dict(checkpoint[name])
    pipeline = SAGEPipeline(config, small, gate, heavy, reflection, TraceEncoder(config))
    mean, std = checkpoint["scaler_mean"].numpy(), checkpoint["scaler_std"].numpy()
    x = torch.tensor((raw-mean)/std, dtype=torch.float32).unsqueeze(0)  # CoLLM eq. 15
    return asdict(pipeline(x, inputs))
