"""Journal-readiness training: complete SAGE pipeline with numerical D2 verifier.

Optimized version with batched processing.
"""
import json, sys, time, hashlib
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sage.config import Config
from sage.data.datasets import PreparedData, prepare_cmapss
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent, confidence_target
from sage.heavy.numerical_heavy import NumericalLarge, NumericalVerifier
from train.training import fingerprint, seed_everything

# Clean test engines (verified: 60 engines not exposed in gate calibration)
with open('runs/gate-final-audit/clean_test_engines.json') as f:
    clean_info = json.load(f)
CLEAN_TEST_ENGINES = clean_info['clean_engines']
LEAKED_ENGINES = clean_info['leaked_engines']

SEEDS = [42, 43, 44, 45, 46]
EPOCHS_SMALL = 20
EPOCHS_GATE = 30
EPOCHS_NUMERICAL = 40
BATCH_SIZE = 256
LR = 0.001
ALPHA = 20.0


def train_small_model(data, config, seed, output_dir):
    """Stage 1: Train small Conv1D-GRU on fit partition only."""
    seed_everything(seed)
    small = SmallModel(config)
    fit_idx = data.indices('fit')
    x = torch.from_numpy(data.x).float()
    y = torch.from_numpy(data.y).float()

    small.target_mean.copy_(y[fit_idx].mean().reshape(1))
    small.target_std.copy_(y[fit_idx].std(unbiased=False).clamp_min(1e-6).reshape(1))

    loader = DataLoader(TensorDataset(x[fit_idx], y[fit_idx]), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(small.parameters(), lr=LR)

    history = []
    for epoch in range(EPOCHS_SMALL):
        small.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            residual = (small(xb).prediction - yb) / small.target_std
            loss = residual.square().mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        history.append({"stage": "small", "epoch": epoch + 1, "scaled_mse": total / len(fit_idx)})
    small.eval().requires_grad_(False)

    torch.save({
        "format_version": 1, "config": asdict(config),
        "small": small.state_dict(),
        "scaler_mean": torch.tensor(data.mean),
        "scaler_std": torch.tensor(data.std),
        "trained": True,
    }, output_dir / "small_model.pt")
    return small, history


def train_gate(data, small, config, seed, output_dir):
    """Stage 2: Train D1 fuzzy gate on confidence partition with engine-balanced loss."""
    seed_everything(seed)
    conf_idx = data.indices('confidence')
    conf_ids = [data.ids[i] for i in conf_idx]
    conf_engine_ids = np.array([int(uid.split(":")[1]) for uid in conf_ids])

    x_conf = torch.from_numpy(data.x[conf_idx]).float()
    y_conf = torch.from_numpy(data.y[conf_idx]).float()

    with torch.inference_mode():
        small_out = small(x_conf)
        qs_targets = confidence_target(small_out.prediction, y_conf, ALPHA)
        conf_features = small_out.features

    gate = FuzzyDecisionAgent(config)
    gate.train()
    opt = torch.optim.Adam(gate.parameters(), lr=LR)

    engine_ids_list = sorted(set(conf_engine_ids.tolist()))
    engine_to_indices = {}
    for i, e in enumerate(conf_engine_ids):
        engine_to_indices.setdefault(int(e), []).append(i)

    history = []
    for epoch in range(EPOCHS_GATE):
        epoch_loss = 0.0
        for e in np.random.permutation(engine_ids_list):
            eng_idx = np.array(engine_to_indices[int(e)])
            eng_x = x_conf[eng_idx]
            eng_t = qs_targets[eng_idx]
            with torch.inference_mode():
                so = small(eng_x)
                feats = so.features
            conf = gate(feats).confidence
            loss = torch.mean((conf - eng_t) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(engine_ids_list)
        history.append({"stage": "gate", "epoch": epoch + 1, "engine_balanced_mse": avg})

    gate.eval()
    torch.save({
        "format_version": 1, "config": asdict(config),
        "small": small.state_dict(), "gate": gate.state_dict(),
        "reflection": None,
        "confidence_examples": len(conf_idx),
        "confidence_ids": conf_ids,
        "confidence_engine_ids": conf_engine_ids.tolist(),
        "gate_training": {
            "loss_type": "engine_balanced_mse",
            "engine_balanced_actually_used": True,
            "n_confidence_windows": len(conf_idx),
            "n_confidence_engines": len(engine_ids_list),
            "seed": seed,
        },
        "trained": True,
    }, output_dir / "sage.pt")
    return gate, history


def train_numerical_heavy_and_verifier(data, config, small, seed, output_dir):
    """Stage 3: Train numerical Patch Transformer heavy + D2 verifier (batched)."""
    seed_everything(seed + 1000)
    fit_idx = data.indices('fit')
    conf_idx = data.indices('confidence')
    x = torch.from_numpy(data.x).float()
    y = torch.from_numpy(data.y).float()

    # Numerical heavy model
    large = NumericalLarge(config.window, config.features, d_model=128, layers=4, heads=4, patch=5)
    large.target_mean.copy_(y[fit_idx].mean().reshape(1))
    large.target_std.copy_(y[fit_idx].std(unbiased=False).clamp_min(1e-6).reshape(1))

    loader = DataLoader(TensorDataset(x[fit_idx], y[fit_idx]), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(large.parameters(), lr=5e-4, weight_decay=1e-4)
    hist_large = []
    for epoch in range(EPOCHS_NUMERICAL):
        large.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred, _ = large(xb)
            loss = ((pred - yb) / large.target_std).square().mean()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        hist_large.append({"stage": "numerical_large", "epoch": epoch + 1, "scaled_mse": total / len(fit_idx)})
    large.eval().requires_grad_(False)

    # D2 verifier: train on confidence-partition numerical heavy traces (batched)
    with torch.inference_mode():
        conf_loader = DataLoader(TensorDataset(x[conf_idx], y[conf_idx]), batch_size=BATCH_SIZE, shuffle=False)
        all_feats = []
        all_preds = []
        all_y = []
        for xb, yb in conf_loader:
            pred, latent = large(xb)
            all_feats.append(latent)
            all_preds.append(pred)
            all_y.append(yb)
        conf_feats = torch.cat(all_feats, dim=0)      # [n, 128]
        conf_preds = torch.cat(all_preds, dim=0)      # [n, 1]
        conf_y = torch.cat(all_y, dim=0)              # [n, 1]
        ql_targets = confidence_target(conf_preds, conf_y, ALPHA)

    verifier = NumericalVerifier(128)
    verifier.train()
    opt = torch.optim.Adam(verifier.parameters(), lr=LR)
    hist_verifier = []
    for epoch in range(EPOCHS_GATE):
        total = 0.0
        loader = DataLoader(TensorDataset(conf_feats, ql_targets), batch_size=BATCH_SIZE, shuffle=True)
        for fb, qb in loader:
            opt.zero_grad()
            loss = ((verifier(fb) - qb) ** 2).sum() / fb.shape[0]
            loss.backward()
            opt.step()
            total += loss.item() * fb.shape[0]
        hist_verifier.append({"stage": "numerical_verifier", "epoch": epoch + 1, "mse": total / len(conf_feats)})
    verifier.eval()

    torch.save({
        "large": large.state_dict(),
        "verifier": verifier.state_dict(),
        "params_large": sum(p.numel() for p in large.parameters()),
        "arch": {"d_model": 128, "layers": 4, "heads": 4, "patch": 5},
        "trained": True,
    }, output_dir / "numeric.pt")

    return large, verifier, hist_large, hist_verifier


def main():
    data_path = Path("data/prepared-fd001")
    data = PreparedData.load(str(data_path))
    config = data.config

    base_dir = Path("runs/journal-training")
    base_dir.mkdir(parents=True, exist_ok=True)

    all_history = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        seed_dir = base_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Small model
        print("\n[Stage 1] Training small Conv1D-GRU on fit partition...")
        small, hist_s = train_small_model(data, config, seed, seed_dir)

        # Stage 2: D1 gate
        print("\n[Stage 2] Training D1 gate on confidence partition (engine-balanced)...")
        gate, hist_g = train_gate(data, small, config, seed, seed_dir)

        # Stage 3: Numerical heavy + D2 verifier
        print("\n[Stage 3] Training numerical heavy + D2 verifier...")
        large, verifier, hist_l, hist_v = train_numerical_heavy_and_verifier(data, config, small, seed, seed_dir)

        all_history[f"seed{seed}"] = {
            "small": hist_s, "gate": hist_g,
            "numerical_large": hist_l, "numerical_verifier": hist_v,
        }

    # Save combined history
    with open(base_dir / "training_history_all.json", "w") as f:
        json.dump(all_history, f, indent=2)
    print(f"\nAll training complete. Artifacts in {base_dir}")


if __name__ == "__main__":
    main()
