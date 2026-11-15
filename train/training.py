"""SAGE training: fit S, freeze S; collect fixed heavy outputs; train F and R.

The API heavy path and text encoder stay frozen. This is a SAGE adaptation of
CoLLM §III-D, not GPT-2 adapter fine-tuning or an exact CoLLM reproduction.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sage.config import Config
from sage.data.datasets import PreparedData
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent, confidence_target
from sage.heavy.nexus_pipeline import NexusPipeline
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder
from sage.pipeline import SAGEPipeline
from sage.trace import TRACE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def fingerprint(data: PreparedData) -> str:
    digest = hashlib.sha256()
    for array in (data.x, data.y, data.mean, data.std, data.split):
        digest.update(array.tobytes())
    digest.update(json.dumps([data.ids, data.timestamps, data.context, data.feature_names, data.target_index, data.config.track, data.config.window, data.config.horizon]).encode())
    return digest.hexdigest()


def train(data: PreparedData, heavy: NexusPipeline, output: str | Path, epochs: int = 20,
          confidence_epochs: int = 30, batch_size: int = 64, confidence_limit: int | None = None,
          narrative: bool = True) -> dict:
    if min(epochs, confidence_epochs, batch_size) < 1 or (confidence_limit is not None and confidence_limit < 1):
        raise ValueError("Epochs, batch size, and optional confidence limit must be positive")
    config = data.config
    if not heavy.backend.identity.startswith("mock") and config.encoder == "hash-demo":
        raise ValueError("Live experiments require the sentence-transformer encoder")
    seed_everything(config.seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    small, gate, reflection = SmallModel(config), FuzzyDecisionAgent(config), SelfReflectionAgent(config)
    encoder = TraceEncoder(config)
    fit_idx, conf_idx = data.indices("fit"), data.indices("confidence")
    if not len(fit_idx) or not len(conf_idx):
        raise ValueError("Fit and confidence partitions must be nonempty")
    x, y = torch.from_numpy(data.x), torch.from_numpy(data.y)
    small.target_mean.copy_(y[fit_idx].mean().reshape(1))
    small.target_std.copy_(y[fit_idx].std(unbiased=False).clamp_min(1e-6).reshape(1))
    loader = DataLoader(TensorDataset(x[fit_idx], y[fit_idx]), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(small.parameters(), lr=config.learning_rate)
    history = []
    TRACE.banner(f"STAGE 1/3  train the small forecaster ({epochs} epochs, {len(fit_idx)} windows)")
    for epoch in range(epochs):
        small.train()
        total = 0.
        for xb, yb in loader:
            optimizer.zero_grad()
            residual = (small(xb).prediction - yb) / small.target_std
            loss = residual.square().mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append({"stage": "small", "epoch": epoch+1, "scaled_mse": total/len(fit_idx)})
        TRACE.epoch("small model", epoch+1, epochs, total/len(fit_idx))
    small.eval().requires_grad_(False)  # CoLLM §III-D hierarchical freezing; SAGE adaptation
    if confidence_limit is not None and confidence_limit < len(conf_idx):
        conf_idx = np.sort(np.random.default_rng(config.seed).choice(conf_idx, confidence_limit, replace=False))
    TRACE.banner(f"STAGE 2/3  run the heavy path on {len(conf_idx)} confidence examples (labels for the gate and verifier)")
    small_features, heavy_features, qs_labels, ql_labels = [], [], [], []
    with (output / "confidence_traces.jsonl").open("w", encoding="utf-8") as traces:
        for number, i in enumerate(conf_idx):
            with torch.no_grad():
                s = small(x[i:i+1])
            TRACE.conf_begin(number+1, len(conf_idx), data.ids[i], y[i].tolist(), s.prediction[0].tolist())
            h = heavy(data.inputs(int(i), narrative=narrative))
            small_features.append(s.features)
            heavy_features.append(encoder(h))
            qs_labels.append(confidence_target(s.prediction, y[i:i+1], config.alpha))
            ql_labels.append(reflection.target(torch.tensor([h.y_l]), y[i:i+1], config.alpha))
            traces.write(json.dumps({"id": data.ids[i], "heavy": asdict(h)}, allow_nan=False)+"\n")
            TRACE.conf_end(h.y_l, y[i].tolist(), qs_labels[-1], ql_labels[-1])
            if (number+1) % 25 == 0:
                print(f"Collected {number+1}/{len(conf_idx)} confidence examples", flush=True)
    TRACE.active = False
    TRACE.banner(f"STAGE 3/3  train the fuzzy gate and the verifier ({confidence_epochs} epochs)")
    conf_data = TensorDataset(torch.cat(small_features), torch.cat(heavy_features), torch.cat(qs_labels), torch.cat(ql_labels))
    optimizer = torch.optim.Adam(list(gate.parameters()) + list(reflection.parameters()), lr=config.learning_rate)
    for epoch in range(confidence_epochs):
        total = 0.
        for sf, hf, qs, ql in DataLoader(conf_data, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = ((gate(sf).confidence-qs)**2).mean() + reflection.loss(reflection(hf), ql)  # CoLLM eq. 14 + §III-B MSE
            loss.backward()
            optimizer.step()
            total += loss.item()*len(sf)
        history.append({"stage": "confidence", "epoch": epoch+1, "joint_mse": total/len(conf_idx)})
        TRACE.epoch("gate + verifier", epoch+1, confidence_epochs, total/len(conf_idx))
    checkpoint = {"format_version": 1, "config": asdict(config), "small": small.state_dict(),
        "gate": gate.state_dict(), "reflection": reflection.state_dict(), "backend": heavy.backend.identity,
        "data_fingerprint": fingerprint(data), "synthetic": data.synthetic, "narrative": narrative,
        "confidence_examples": len(conf_idx), "confidence_ids": [data.ids[i] for i in conf_idx],
        "trained": True, "torch_version": str(torch.__version__),
        "scaler_mean": torch.tensor(data.mean), "scaler_std": torch.tensor(data.std),
        "feature_names": data.feature_names, "target_index": data.target_index,
        # Prompt and experiment metadata
        "prompt_version": heavy.prompt_version,
        "prompt_hash": hashlib.sha256(json.dumps(heavy._prompts, sort_keys=True).encode()).hexdigest(),
        "effective_agent_prompts": heavy._prompts,
        "effective_task_instruction": heavy._task_instruction,
        "backend_identity": heavy.backend.identity,
        "dataset": data.config.track,
        "splits": {"fit": len(data.indices("fit")), "confidence": len(data.indices("confidence")),
                   "validation": len(data.indices("validation")), "test": len(data.indices("test"))},
        "random_seed": config.seed,
        "checkpoint_identity": str(output / "sage.pt")}
    torch.save(checkpoint, output / "sage.pt")
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"checkpoint": str(output / "sage.pt"), "fit_examples": len(fit_idx),
            "confidence_examples": len(conf_idx), "synthetic": data.synthetic, "backend": heavy.backend.identity}


def load_pipeline(path: str | Path, data: PreparedData, heavy: NexusPipeline,
                  prompt_version: str | None = None) -> tuple[SAGEPipeline, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not checkpoint.get("trained") or checkpoint["format_version"] != 1:
        raise ValueError("A trained compatible checkpoint is required")
    if checkpoint["data_fingerprint"] != fingerprint(data):
        raise ValueError("Prepared data differs from the checkpoint's experiment")
    if checkpoint["backend"] != heavy.backend.identity:
        raise ValueError("Heavy backend differs from confidence training; retrain F/R")
    # Resolve prompt version: explicit override > recorded > legacy default v1
    recorded_pv = checkpoint.get("prompt_version", "v1")
    if prompt_version is None:
        resolved_pv = recorded_pv
    elif prompt_version != recorded_pv:
        raise ValueError(
            f"Prompt version mismatch: checkpoint recorded '{recorded_pv}', "
            f"requested '{prompt_version}'. Use --prompt-version {recorded_pv} "
            "or pass the matching version explicitly.")
    else:
        resolved_pv = prompt_version
    if resolved_pv != heavy.prompt_version:
        raise ValueError(
            f"NexusPipeline prompt_version '{heavy.prompt_version}' does not match "
            f"resolved checkpoint prompt_version '{resolved_pv}'. Reconstruct NexusPipeline "
            f"with prompt_version='{resolved_pv}'.")
    config = Config(**checkpoint["config"])
    config = replace(config, prompt_version=resolved_pv)
    small, gate, reflection = SmallModel(config), FuzzyDecisionAgent(config), SelfReflectionAgent(config)
    for name, model in [("small", small), ("gate", gate), ("reflection", reflection)]:
        model.load_state_dict(checkpoint[name])
    return SAGEPipeline(config, small, gate, heavy, reflection, TraceEncoder(config)), checkpoint
