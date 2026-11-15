"""Journal-readiness evaluation: evaluate SAGE variants on clean test engines.

Evaluates on verified-clean 60 test engines only.
Computes RMSE, MAE, NASA score, escalation, coverage, engine-clustered bootstrap CIs.
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

# Clean test engines
with open('runs/gate-final-audit/clean_test_engines.json') as f:
    clean_info = json.load(f)
CLEAN_TEST_ENGINES = clean_info['clean_engines']

SEEDS = [42, 43, 44, 45, 46]
ALPHA = 20.0
BATCH_SIZE = 256


def nasa_score(pred, truth):
    d = np.asarray(pred, float) - np.asarray(truth, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def rmse(pred, truth):
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(truth, float)) ** 2)))


def mae(pred, truth):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(truth, float))))


def bootstrap_engine_ci(pred, truth, engine_ids, seed=42, n=2000):
    """Engine-clustered bootstrap CI for RMSE."""
    rng = np.random.default_rng(seed)
    unique_engines = np.unique(engine_ids)
    engine_to_idx = {e: np.where(engine_ids == e)[0] for e in unique_engines}
    vals = []
    for _ in range(n):
        sampled_engines = rng.choice(unique_engines, size=len(unique_engines), replace=True)
        boot_idx = np.concatenate([engine_to_idx[int(e)] for e in sampled_engines])
        vals.append(np.sqrt(((pred[boot_idx] - truth[boot_idx]) ** 2).mean()))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_small_and_gate(seed, training_dir):
    """Load small model and gate from training artifacts."""
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
    """Load numerical heavy + verifier."""
    seed_dir = training_dir / f"seed{seed}"
    ckpt = torch.load(seed_dir / "numeric.pt", map_location="cpu", weights_only=True)
    large = NumericalLarge(50, 14, d_model=128, layers=4, heads=4, patch=5)
    verifier = NumericalVerifier(128)
    large.load_state_dict(ckpt["large"])
    verifier.load_state_dict(ckpt["verifier"])
    large.eval()
    verifier.eval()
    return large, verifier


def evaluate_variant(data, test_idx, small, gate, large, verifier, config, mode, tau1, tau2):
    """Evaluate a single variant on test indices."""
    x = torch.from_numpy(data.x).float()
    y = data.y.ravel().astype(float)
    ids = data.ids

    predictions = []
    sources = []
    q_s_vals = []
    q_l_vals = []
    d1_vals = []

    with torch.inference_mode():
        for i in test_idx:
            xi = x[i:i+1]
            small_out = small(xi)
            y_s = small_out.prediction.item()
            q_s = gate(small_out.features).confidence.item()
            d1 = q_s >= tau1
            y_true = y[i]

            if mode == "small_only":
                pred = y_s
                src = "small"
                q_l = None; d2 = None
            elif mode == "numerical_heavy":
                pred_l, latent = large(xi)
                y_l = pred_l.item()
                q_l = verifier(latent).item()
                delta = q_s - q_l
                d2 = delta <= tau2
                if d2:
                    pred = y_l
                    src = "heavy"
                else:
                    pred = (y_s + y_l) / 2
                    src = "fusion"
            elif mode == "numerical_sage":
                if d1:
                    pred = y_s
                    src = "small"
                    q_l = None; d2 = None
                else:
                    pred_l, latent = large(xi)
                    y_l = pred_l.item()
                    q_l = verifier(latent).item()
                    delta = q_s - q_l
                    d2 = delta <= tau2
                    if d2:
                        pred = y_l
                        src = "heavy"
                    else:
                        pred = (y_s + y_l) / 2
                        src = "fusion"
            elif mode == "always_average":
                pred_l, _ = large(xi)
                y_l = pred_l.item()
                pred = (y_s + y_l) / 2
                src = "fusion"
                q_l = None; d2 = None
            elif mode == "sage_weighted":
                if d1:
                    pred = y_s
                    src = "small"
                    q_l = None; d2 = None
                else:
                    pred_l, latent = large(xi)
                    y_l = pred_l.item()
                    q_l = verifier(latent).item()
                    delta = q_s - q_l
                    d2 = delta <= tau2
                    if d2:
                        pred = y_l
                        src = "heavy"
                    else:
                        w = q_s / (q_s + q_l) if (q_s + q_l) > 0 else 0.5
                        pred = w * y_s + (1 - w) * y_l
                        src = "fusion"

            predictions.append(pred)
            sources.append(src)
            q_s_vals.append(q_s)
            if q_l is not None:
                q_l_vals.append(q_l)
            d1_vals.append(d1)

    predictions = np.array(predictions)
    truth = y[test_idx]
    engine_ids = np.array([int(ids[i].split(":")[1]) for i in test_idx])

    n_heavy = sum(1 for s in sources if s == "heavy")
    n_fusion = sum(1 for s in sources if s == "fusion")
    n_small = sum(1 for s in sources if s == "small")

    return {
        "rmse": rmse(predictions, truth),
        "mae": mae(predictions, truth),
        "nasa_score": nasa_score(predictions, truth),
        "n_samples": len(predictions),
        "heavy_call_percent": 100.0 * n_heavy / len(predictions),
        "fusion_percent": 100.0 * n_fusion / len(predictions),
        "small_percent": 100.0 * n_small / len(predictions),
        "escalation_rate_percent": 100.0 * (1 - np.mean(d1_vals)),
        "coverage_percent": 100.0 * np.mean(d1_vals),
        "mean_q_s": float(np.mean(q_s_vals)),
        "mean_q_l": float(np.mean(q_l_vals)) if q_l_vals else None,
        "engine_ids": engine_ids.tolist(),
        "predictions": predictions.tolist(),
        "truth": truth.tolist(),
        "sources": sources,
    }


def main():
    data_path = Path("data/prepared-fd001")
    data = PreparedData.load(str(data_path))
    config = data.config

    # Get clean test engine indices
    test_idx_all = data.indices("test")
    test_idx_clean = [i for i in test_idx_all if int(data.ids[i].split(":")[1]) in CLEAN_TEST_ENGINES]
    print(f"Clean test engines: {len(CLEAN_TEST_ENGINES)}")
    print(f"Clean test windows: {len(test_idx_clean)}")

    training_dir = Path("runs/journal-training")
    eval_dir = Path("runs/journal-eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    modes = ["small_only", "numerical_heavy", "numerical_sage", "always_average", "sage_weighted"]

    all_results = {}
    all_predictions = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        small, gate, cfg = load_small_and_gate(seed, training_dir)
        large, verifier = load_numerical(seed, training_dir)

        seed_results = {}
        for mode in modes:
            res = evaluate_variant(data, test_idx_clean, small, gate, large, verifier, cfg,
                                   mode, cfg.tau1, cfg.tau2)
            # Add bootstrap CI
            ci = bootstrap_engine_ci(np.array(res["predictions"]), np.array(res["truth"]),
                                     np.array(res["engine_ids"]), seed=seed)
            res["rmse_ci95"] = ci
            seed_results[mode] = res
            all_predictions.append({
                "seed": seed, "mode": mode, **{k: v for k, v in res.items()
                    if k not in ["engine_ids", "predictions", "truth", "sources"]}
            })
            print(f"  {mode}: RMSE={res['rmse']:.4f} MAE={res['mae']:.4f} "
                  f"NASA={res['nasa_score']:.1f} heavy%={res['heavy_call_percent']:.1f} "
                  f"escalation%={res['escalation_rate_percent']:.1f}")

        all_results[f"seed{seed}"] = seed_results

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS SEEDS")
    print(f"{'='*60}")
    aggregate = {}
    for mode in modes:
        rmses = [all_results[f"seed{s}"][mode]["rmse"] for s in SEEDS]
        maes = [all_results[f"seed{s}"][mode]["mae"] for s in SEEDS]
        nasas = [all_results[f"seed{s}"][mode]["nasa_score"] for s in SEEDS]
        escs = [all_results[f"seed{s}"][mode]["escalation_rate_percent"] for s in SEEDS]
        aggregate[mode] = {
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "nasa_mean": float(np.mean(nasas)), "nasa_std": float(np.std(nasas)),
            "escalation_mean": float(np.mean(escs)), "escalation_std": float(np.std(escs)),
            "rmse_ci95_bootstrap_across_seeds": [float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))],
        }
        print(f"  {mode}: RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} "
              f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} "
              f"NASA={np.mean(nasas):.1f}±{np.std(nasas):.1f}")

    # Save results
    with open(eval_dir / "results.json", "w") as f:
        json.dump({"aggregate": aggregate, "per_seed": all_results}, f, indent=2)
    with open(eval_dir / "predictions_summary.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    # Save per-sample predictions for seed 43 (primary)
    primary = all_results["seed43"]
    with open(eval_dir / "per_sample_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["engine", "cycle", "window_idx", "variant", "prediction", "truth", "source", "q_s"])
        for mode in modes:
            res = primary[mode]
            for j, i in enumerate(test_idx_clean):
                uid = data.ids[i]
                writer.writerow([
                    uid.split(":")[1], uid.split(":")[2], i, mode,
                    f"{res['predictions'][j]:.6f}", f"{res['truth'][j]:.6f}",
                    res["sources"][j], f"{res['mean_q_s']:.6f}"
                ])

    print(f"\nResults saved to {eval_dir}")


if __name__ == "__main__":
    from sage.heavy.numerical_heavy import NumericalLarge, NumericalVerifier
    main()
