"""Threshold tuning on validation data only (never touches test labels).

Grid searches tau1 and tau2 for numerical_sage mode, selects best pair
by validation RMSE, then applies to clean test set.
"""
import json, csv
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sage.config import Config
from sage.data.datasets import PreparedData
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent, confidence_target
from sage.heavy.numerical_heavy import NumericalLarge, NumericalVerifier
from sage.verify.decision import decide
from sage.heavy.numerical_heavy import NumericalLarge, NumericalVerifier

with open('runs/gate-final-audit/clean_test_engines.json') as f:
    clean_info = json.load(f)
CLEAN_TEST_ENGINES = clean_info['clean_engines']

SEEDS = [42, 43, 44, 45, 46]
ALPHA = 20.0
BATCH_SIZE = 256

TAU1_CANDIDATES = [0.15, 0.30, 0.40, 0.60, 0.80]
TAU2_CANDIDATES = [0.0, 0.05, 0.10, 0.20]


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(truth, float)) ** 2)))


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(truth, float))))


def nasa_score(pred, truth):
    d = np.asarray(pred, float) - np.asarray(truth, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def load_small_and_gate(seed, training_dir):
    seed_dir = training_dir / f"seed{seed}"
    ckpt = torch.load(seed_dir / "sage.pt", map_location="cpu", weights_only=True)
    config = Config(**ckpt["config"])
    small = SmallModel(config)
    gate = FuzzyDecisionAgent(config)
    small.load_state_dict(ckpt["small"])
    gate.load_state_dict(ckpt["gate"])
    small.eval()
    gate.eval()
    return small, gate, config


def load_numerical(seed, training_dir):
    seed_dir = training_dir / f"seed{seed}"
    ckpt = torch.load(seed_dir / "numeric.pt", map_location="cpu", weights_only=True)
    large = NumericalLarge(50, 14, d_model=128, layers=4, heads=4, patch=5)
    verifier = NumericalVerifier(128)
    large.load_state_dict(ckpt["large"])
    verifier.load_state_dict(ckpt["verifier"])
    large.eval()
    verifier.eval()
    return large, verifier


def compute_validation_outputs(data, val_idx, small, gate, large, verifier, config):
    """Compute small predictions, gate confidences, heavy predictions, verifier confidences on validation."""
    x = torch.from_numpy(data.x).float()
    y = data.y.ravel().astype(float)
    ids = data.ids

    y_s_vals = []
    q_s_vals = []
    y_l_vals = []
    q_l_vals = []
    y_true_vals = []

    with torch.inference_mode():
        for i in val_idx:
            xi = x[i:i+1]
            small_out = small(xi)
            y_s = small_out.prediction.item()
            q_s = gate(small_out.features).confidence.item()
            pred_l, latent = large(xi)
            y_l = pred_l.item()
            q_l = verifier(latent).item()

            y_s_vals.append(y_s)
            q_s_vals.append(q_s)
            y_l_vals.append(y_l)
            q_l_vals.append(q_l)
            y_true_vals.append(y[i])

    return (np.array(y_s_vals), np.array(q_s_vals), np.array(y_l_vals),
            np.array(q_l_vals), np.array(y_true_vals))


def evaluate_numerical_sage_with_thresholds(y_s, q_s, y_l, q_l, y_true, tau1, tau2):
    """Evaluate numerical_sage with given thresholds."""
    predictions = []
    sources = []
    for i in range(len(y_true)):
        if q_s[i] >= tau1:
            predictions.append(y_s[i])
            sources.append("small")
        else:
            delta = q_s[i] - q_l[i]
            if delta <= tau2:
                predictions.append(y_l[i])
                sources.append("heavy")
            else:
                predictions.append((y_s[i] + y_l[i]) / 2)
                sources.append("fusion")

    predictions = np.array(predictions)
    return {
        "rmse": rmse(predictions, y_true),
        "mae": mae(predictions, y_true),
        "nasa": nasa_score(predictions, y_true),
        "heavy_percent": 100.0 * sum(1 for s in sources if s == "heavy") / len(predictions),
        "fusion_percent": 100.0 * sum(1 for s in sources if s == "fusion") / len(predictions),
        "small_percent": 100.0 * sum(1 for s in sources if s == "small") / len(predictions),
        "predictions": predictions,
        "sources": sources,
    }


def main():
    data_path = Path("data/prepared-fd001")
    data = PreparedData.load(str(data_path))
    config = data.config

    # Validation indices
    val_idx = data.indices("validation")

    # Clean test indices
    test_idx_all = data.indices("test")
    test_idx_clean = [i for i in test_idx_all if int(data.ids[i].split(":")[1]) in CLEAN_TEST_ENGINES]

    training_dir = Path("runs/journal-training")
    tune_dir = Path("runs/journal-tuning")
    tune_dir.mkdir(parents=True, exist_ok=True)

    all_tuning_results = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed} - Threshold Tuning")
        print(f"{'='*60}")
        small, gate, cfg = load_small_and_gate(seed, training_dir)
        large, verifier = load_numerical(seed, training_dir)

        # Compute validation outputs
        y_s_val, q_s_val, y_l_val, q_l_val, y_true_val = compute_validation_outputs(
            data, val_idx, small, gate, large, verifier, config)

        # Grid search
        best_tau1, best_tau2 = cfg.tau1, cfg.tau2
        best_rmse = float('inf')
        tuning_grid = []

        for t1 in TAU1_CANDIDATES:
            for t2 in TAU2_CANDIDATES:
                res = evaluate_numerical_sage_with_thresholds(
                    y_s_val, q_s_val, y_l_val, q_l_val, y_true_val, t1, t2)
                tuning_result = {
                    "tau1": t1, "tau2": t2,
                    "val_rmse": res["rmse"], "val_mae": res["mae"],
                    "val_nasa": res["nasa"],
                    "heavy_percent": res["heavy_percent"],
                    "fusion_percent": res["fusion_percent"],
                    "small_percent": res["small_percent"],
                }
                tuning_grid.append(tuning_result)
                if res["rmse"] < best_rmse:
                    best_rmse = res["rmse"]
                    best_tau1, best_tau2 = t1, t2

        print(f"  Best thresholds: tau1={best_tau1}, tau2={best_tau2}, val_RMSE={best_rmse:.4f}")
        print(f"  Default (0.60, 0.05) val_RMSE: {rmse(evaluate_numerical_sage_with_thresholds(y_s_val, q_s_val, y_l_val, q_l_val, y_true_val, 0.60, 0.05)['predictions'], y_true_val):.4f}")

        # Apply tuned thresholds to clean test set
        y_s_test, q_s_test, y_l_test, q_l_test, y_true_test = compute_validation_outputs(
            data, test_idx_clean, small, gate, large, verifier, config)

        tuned_res = evaluate_numerical_sage_with_thresholds(
            y_s_test, q_s_test, y_l_test, q_l_test, y_true_test, best_tau1, best_tau2)
        default_res = evaluate_numerical_sage_with_thresholds(
            y_s_test, q_s_test, y_l_test, q_l_test, y_true_test, 0.60, 0.05)

        print(f"  Test (tuned {best_tau1}, {best_tau2}): RMSE={tuned_res['rmse']:.4f} MAE={tuned_res['mae']:.4f} NASA={tuned_res['nasa']:.1f}")
        print(f"  Test (default 0.60, 0.05): RMSE={default_res['rmse']:.4f} MAE={default_res['mae']:.4f} NASA={default_res['nasa']:.1f}")

        all_tuning_results[f"seed{seed}"] = {
            "best_tau1": best_tau1, "best_tau2": best_tau2, "best_val_rmse": best_rmse,
            "tuning_grid": tuning_grid,
            "test_tuned": {k: v for k, v in tuned_res.items() if k not in ["predictions", "sources"]},
            "test_default": {k: v for k, v in default_res.items() if k not in ["predictions", "sources"]},
        }

    # Save tuning results
    with open(tune_dir / "threshold_tuning.json", "w") as f:
        json.dump(all_tuning_results, f, indent=2)

    # Aggregate
    print(f"\n{'='*60}")
    print("THRESHOLD TUNING AGGREGATE")
    print(f"{'='*60}")
    best_tau1s = [all_tuning_results[f"seed{s}"]["best_tau1"] for s in SEEDS]
    best_tau2s = [all_tuning_results[f"seed{s}"]["best_tau2"] for s in SEEDS]
    print(f"  Best tau1 values: {best_tau1s}")
    print(f"  Best tau2 values: {best_tau2s}")
    print(f"  Most common tau1: {max(set(best_tau1s), key=best_tau1s.count)}")
    print(f"  Most common tau2: {max(set(best_tau2s), key=best_tau2s.count)}")

    tuned_rmses = [all_tuning_results[f"seed{s}"]["test_tuned"]["rmse"] for s in SEEDS]
    default_rmses = [all_tuning_results[f"seed{s}"]["test_default"]["rmse"] for s in SEEDS]
    print(f"  Test tuned RMSE: {np.mean(tuned_rmses):.4f}±{np.std(tuned_rmses):.4f}")
    print(f"  Test default RMSE: {np.mean(default_rmses):.4f}±{np.std(default_rmses):.4f}")

    print(f"\nResults saved to {tune_dir}")


if __name__ == "__main__":
    main()
