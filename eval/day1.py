"""Day-1 FD001 pipeline: trains the non-LLM components and evaluates the seven variants on identical samples.

Variants: small_only, heavy_only, gate_only, always_average, sage_average, sage_weighted, numerical_heavy.
LLM-dependent variants need a live backend + a checkpoint produced by `main.py train --backend openai`.
Nothing here fabricates or imputes predictions: a variant that cannot run is reported as NOT RUN.

  python -m eval.day1 train-base --data data/prepared-fd001 --output runs/fd001-day1 --confidence-limit 200
  python -m eval.day1 evaluate   --data data/prepared-fd001 --base runs/fd001-day1/base.pt \
        --numeric runs/fd001-day1/numeric.pt --output results/fd001-day1 [--backend openai --checkpoint runs/fd001-day1/sage.pt]
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sage.config import Config
from sage.data.datasets import PreparedData
from sage.forecaster.small_model import SmallModel
from sage.gate.fuzzy_gate import FuzzyDecisionAgent, confidence_target
from sage.heavy.nexus_pipeline import MockBackend, NexusPipeline
from sage.heavy.numerical_heavy import NumericalLarge, NumericalVerifier
from sage.pipeline import SAGEPipeline
from sage.verify.decision import decide
from sage.verify.self_reflection import SelfReflectionAgent, TraceEncoder
from train.training import fingerprint, load_pipeline, seed_everything

log = logging.getLogger("day1")
VARIANTS = ["small_only", "heavy_only", "gate_only", "always_average", "sage_average", "sage_weighted", "numerical_heavy"]
LLM_VARIANTS = ["heavy_only", "gate_only", "always_average", "sage_average", "sage_weighted"]
# variant -> (pipeline mode, fusion)
PIPELINE_MODE = {"heavy_only": ("heavy_always", "average"), "gate_only": ("gate_only", "average"),
                 "always_average": ("always_average", "average"), "sage_average": ("sage", "average"),
                 "sage_weighted": ("sage", "confidence_weighted")}


# ------------------------------------------------------------------ metrics
def nasa_score(pred: np.ndarray, truth: np.ndarray) -> float:
    d = np.asarray(pred, float) - np.asarray(truth, float)  # PHM08 score; late predictions (d>0) penalised harder
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def rmse(pred, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(truth, float)) ** 2)))


def bootstrap_rmse_ci(pred, truth, seed: int = 42, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    idx = rng.integers(0, len(p), size=(n, len(p)))
    vals = np.sqrt(((p[idx] - t[idx]) ** 2).mean(axis=1))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------ training (non-LLM components)
def train_base(data: PreparedData, out: Path, epochs: int, conf_epochs: int, conf_limit: int | None,
               numeric_epochs: int, batch_size: int = 64) -> dict:
    # Same config as the live CLI (`main.py train --backend openai`) so small/gate are comparable with sage.pt.
    config = replace(data.config, encoder="sentence-transformer", embedding_dim=384)
    seed_everything(config.seed)
    out.mkdir(parents=True, exist_ok=True)
    small, gate, reflection = SmallModel(config), FuzzyDecisionAgent(config), SelfReflectionAgent(config)  # RNG order == train()
    fit_idx, conf_idx = data.indices("fit"), data.indices("confidence")
    x, y = torch.from_numpy(data.x), torch.from_numpy(data.y)
    small.target_mean.copy_(y[fit_idx].mean().reshape(1))
    small.target_std.copy_(y[fit_idx].std(unbiased=False).clamp_min(1e-6).reshape(1))
    loader = DataLoader(TensorDataset(x[fit_idx], y[fit_idx]), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(small.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(epochs):
        small.train(); total = 0.
        for xb, yb in loader:
            opt.zero_grad()
            loss = ((small(xb).prediction - yb) / small.target_std).square().mean()
            loss.backward(); opt.step(); total += loss.item() * len(xb)
        history.append({"stage": "small", "epoch": epoch + 1, "scaled_mse": total / len(fit_idx)})
    small.eval().requires_grad_(False)
    if conf_limit is not None and conf_limit < len(conf_idx):  # identical subset rule to train()
        conf_idx = np.sort(np.random.default_rng(config.seed).choice(conf_idx, conf_limit, replace=False))
    with torch.no_grad():
        feats = torch.cat([small(x[i:i + 1]).features for i in conf_idx])
        preds = torch.cat([small(x[i:i + 1]).prediction for i in conf_idx])
    qs = confidence_target(preds, y[conf_idx], config.alpha)
    opt = torch.optim.Adam(gate.parameters(), lr=config.learning_rate)  # gate loss is separable from verifier loss
    for epoch in range(conf_epochs):
        total = 0.
        for fb, qb in DataLoader(TensorDataset(feats, qs), batch_size=batch_size, shuffle=True):
            opt.zero_grad(); loss = ((gate(fb).confidence - qb) ** 2).mean(); loss.backward(); opt.step()
            total += loss.item() * len(fb)
        history.append({"stage": "gate", "epoch": epoch + 1, "mse": total / len(conf_idx)})
    gate.eval()
    torch.save({"format_version": 1, "kind": "base", "config": asdict(config), "small": small.state_dict(),
                "gate": gate.state_dict(), "data_fingerprint": fingerprint(data), "confidence_ids": [data.ids[i] for i in conf_idx],
                "confidence_examples": len(conf_idx), "trained": True, "torch_version": str(torch.__version__),
                "scaler_mean": torch.tensor(data.mean), "scaler_std": torch.tensor(data.std)}, out / "base.pt")

    # numerical large model: fit partition only; verifier on confidence subset (same budget as gate)
    torch.manual_seed(config.seed + 1000)
    large, verifier = NumericalLarge(config.window, config.features), NumericalVerifier(128)
    large.target_mean.copy_(y[fit_idx].mean().reshape(1)); large.target_std.copy_(y[fit_idx].std(unbiased=False).clamp_min(1e-6).reshape(1))
    opt = torch.optim.Adam(large.parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x[fit_idx], y[fit_idx]), batch_size=batch_size, shuffle=True)
    for epoch in range(numeric_epochs):
        large.train(); total = 0.
        for xb, yb in loader:
            opt.zero_grad(); loss = ((large(xb)[0] - yb) / large.target_std).square().mean(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        history.append({"stage": "numerical_large", "epoch": epoch + 1, "scaled_mse": total / len(fit_idx)})
    large.eval().requires_grad_(False)
    with torch.no_grad():
        lat = torch.cat([large(x[i:i + 1])[1] for i in conf_idx]); lp = torch.cat([large(x[i:i + 1])[0] for i in conf_idx])
        val = data.indices("validation"); vp = torch.cat([large(x[val[j:j + 512]])[0] for j in range(0, len(val), 512)])
        sv = torch.cat([small(x[val[j:j + 512]]).prediction for j in range(0, len(val), 512)])
    ql = confidence_target(lp, y[conf_idx], config.alpha)
    opt = torch.optim.Adam(verifier.parameters(), lr=config.learning_rate)
    for epoch in range(conf_epochs):
        total = 0.
        for lb, qb in DataLoader(TensorDataset(lat, ql), batch_size=batch_size, shuffle=True):
            opt.zero_grad(); loss = ((verifier(lb) - qb) ** 2).mean(); loss.backward(); opt.step(); total += loss.item() * len(lb)
        history.append({"stage": "numerical_verifier", "epoch": epoch + 1, "mse": total / len(conf_idx)})
    torch.save({"large": large.state_dict(), "verifier": verifier.state_dict(), "data_fingerprint": fingerprint(data),
                "params_large": sum(p.numel() for p in large.parameters()), "arch": {"d_model": 128, "layers": 4, "heads": 4, "patch": 5}},
               out / "numeric.pt")
    info = {"fit_windows": len(fit_idx), "confidence_examples": len(conf_idx),
            "validation_rmse_small": rmse(sv.numpy(), y[val].numpy()), "validation_rmse_numerical_large": rmse(vp.numpy(), y[val].numpy()),
            "params_small": sum(p.numel() for p in small.parameters()), "params_large": sum(p.numel() for p in large.parameters())}
    (out / "training_history.json").write_text(json.dumps({"summary": info, "history": history}, indent=1))
    return info


# ------------------------------------------------------------------ evaluation
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(path: Path, config_override: Config | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    config = Config(**ck["config"])
    small, gate = SmallModel(config), FuzzyDecisionAgent(config)
    small.load_state_dict(ck["small"]); gate.load_state_dict(ck["gate"])
    return ck, config, small.eval(), gate.eval()


def row(vname, data, i, res: dict, truth_off: float) -> dict:
    return {"variant": vname, "id": data.ids[i], "engine": int(data.ids[i].split(":")[1]), "true_rul_capped": float(data.y[i][0]),
            "true_rul_official": truth_off, **res}


def evaluate(args) -> dict:
    out = Path(args.output); (out / "predictions").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    data = PreparedData.load(args.data)
    test = data.indices("test")
    if len(test) != 100 or data.config.track != "FD001":
        raise ValueError("Day-1 protocol: FD001 official test set (100 engines) required")
    truth_off = np.loadtxt(Path(args.rul)).reshape(-1)  # official, uncapped
    base_ck, config, small, gate = load_models(Path(args.base))
    if base_ck["data_fingerprint"] != fingerprint(data):
        raise ValueError("Prepared data differs from the base checkpoint's data")
    numeric = torch.load(args.numeric, map_location="cpu", weights_only=True) if args.numeric else None
    requested = args.variants or VARIANTS
    status, rows, reports = {}, [], {}
    narrative = True
    heavy_pipe = pipeline = sage_ck = None
    if args.backend:
        if not args.checkpoint:
            raise ValueError("--checkpoint (sage.pt from `main.py train --backend ...`) is required with --backend")
        from sage.cli import backend as make_backend
        probe = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = Config(**probe["config"])
        wiring = args.backend == "mock"
        if wiring and not args.wiring_test:
            raise ValueError("--backend mock only with --wiring-test; mock output is never a research result")
        heavy_pipe = make_backend(args.backend, cfg, Path(args.checkpoint).parent / "cache",
                                  prompt_version=cfg.prompt_version)
        pipeline, sage_ck = load_pipeline(args.checkpoint, data, heavy_pipe)
        narrative = sage_ck["narrative"]
        same_small = all(torch.equal(sage_ck["small"][k], base_ck["small"][k]) for k in base_ck["small"])
        same_gate = all(torch.equal(sage_ck["gate"][k], base_ck["gate"][k]) for k in base_ck["gate"])
        log.info("small identical between base.pt and sage.pt: %s | gate identical: %s", same_small, same_gate)
        if not same_small and not args.wiring_test:  # one model must serve all variants
            raise ValueError("base.pt and sage.pt contain different small models; retrain with identical seed/config")
        small, gate = pipeline.small, pipeline.gate  # all variants use the checkpoint evaluated for LLM variants
        config = pipeline.config
    if numeric is not None:
        large = NumericalLarge(config.window, config.features); large.load_state_dict(numeric["large"]); large.eval()
        ver = NumericalVerifier(128); ver.load_state_dict(numeric["verifier"]); ver.eval()
    for v in requested:
        if v in LLM_VARIANTS and pipeline is None:
            status[v] = "NOT RUN: needs live LLM backend (--backend openai) and trained verifier checkpoint"; continue
        if v == "numerical_heavy" and numeric is None:
            status[v] = "NOT RUN: numeric.pt missing"; continue
        t_wall = time.perf_counter(); recs = []
        for k, i in enumerate(test, 1):
            xi = torch.from_numpy(data.x[i:i + 1]);
            with torch.inference_mode():
                if v == "small_only":
                    s0 = time.perf_counter(); y_s = small(xi).prediction[0].tolist(); lat = time.perf_counter() - s0
                    res = dict(pred=y_s[0], source="small", q_s=None, q_l=None, delta=None, d1=None, d2=None, heavy_called=0,
                               latency_s=lat, input_tokens=0, output_tokens=0, cache_hit=None, y_small=y_s[0], y_heavy=None)
                elif v == "numerical_heavy":
                    c0 = time.perf_counter(); sm = small(xi); g = gate(sm.features)
                    y_s, q_s, d1 = sm.prediction[0].tolist(), g.confidence.item(), bool(g.accept_small.item())
                    if d1:
                        pred, src, q_l, delta, d2, y_l = y_s[0], "small", None, None, None, None
                    else:
                        pl, lat_feat = large(xi); y_l = pl[0].tolist(); q_l = ver(lat_feat).item()
                        dec = decide(y_s, y_l, q_s, q_l, replace(config, fusion="average"))
                        pred, src, delta, d2, y_l = dec.prediction[0], dec.source, dec.delta, dec.accept_heavy, y_l[0]
                    res = dict(pred=pred, source=src, q_s=q_s, q_l=q_l, delta=delta, d1=d1, d2=d2, heavy_called=0 if d1 else 1,
                               latency_s=time.perf_counter() - c0, input_tokens=0, output_tokens=0, cache_hit=None, y_small=y_s[0], y_heavy=y_l)
                else:
                    mode, fusion = PIPELINE_MODE[v]
                    pipeline.config = replace(config, fusion=fusion)
                    r = pipeline(xi, data.inputs(int(i), narrative), mode)
                    toks = [0, 0]
                    if r.heavy is not None:
                        for tr in (r.heavy.context, r.heavy.macro, r.heavy.micro, r.heavy.synthesis):
                            toks[0] += tr.input_tokens; toks[1] += tr.output_tokens
                    live = r.elapsed_seconds + (r.heavy.elapsed_seconds if r.heavy is not None and r.heavy.cache_hit else 0.0)
                    res = dict(pred=r.prediction[0], source=r.source, q_s=r.q_s, q_l=r.q_l, delta=r.delta, d1=r.d1, d2=r.d2,
                               heavy_called=int(r.heavy is not None), latency_s=live, input_tokens=toks[0], output_tokens=toks[1],
                               cache_hit=None if r.heavy is None else r.heavy.cache_hit, y_small=r.y_s[0],
                               y_heavy=None if r.heavy is None else r.heavy.y_l[0])
            recs.append(row(v, data, i, res, float(truth_off[int(data.ids[i].split(":")[1]) - 1])))
        df = pd.DataFrame(recs)
        pred, tc, to = df.pred.to_numpy(), df.true_rul_capped.to_numpy(), df.true_rul_official.to_numpy()
        p_in, p_out = os.getenv("SAGE_PRICE_IN_PER_M"), os.getenv("SAGE_PRICE_OUT_PER_M")
        cost = None if not (p_in and p_out) else (df.input_tokens.sum() * float(p_in) + df.output_tokens.sum() * float(p_out)) / 1e6
        rep = {"n": len(df), "rmse": rmse(pred, tc), "rmse_ci95_bootstrap": bootstrap_rmse_ci(pred, tc), "mae": float(np.mean(np.abs(pred - tc))),
               "nasa_score": nasa_score(pred, tc), "rmse_vs_official_uncapped_rul": rmse(pred, to), "nasa_score_vs_official_uncapped_rul": nasa_score(pred, to),
               "latency_mean_s": float(df.latency_s.mean()), "latency_p50_s": float(df.latency_s.median()), "latency_p95_s": float(df.latency_s.quantile(.95)),
               "latency_total_s": float(df.latency_s.sum()), "heavy_call_percent": float(100 * df.heavy_called.mean()),
               "final_small_percent": float(100 * (df.source == "small").mean()), "final_heavy_percent": float(100 * (df.source == "heavy").mean()),
               "final_fusion_percent": float(100 * (df.source == "fusion").mean()),
               "agent_calls": int(4 * df.heavy_called.sum()) if v in LLM_VARIANTS else 0,
               "input_tokens": int(df.input_tokens.sum()), "output_tokens": int(df.output_tokens.sum()), "est_cost_usd": cost,
               "cost_note": None if v in LLM_VARIANTS else "no API cost; numerical compute only",
               "wall_seconds_this_run": time.perf_counter() - t_wall}
        if v == "numerical_heavy":
            rep["params_heavy"] = int(numeric["params_large"])
        reports[v] = rep; status[v] = "RUN"
        df.to_csv(out / "predictions" / f"{v}.csv", index=False); rows.append(df)
        log.info("%-16s RMSE %.3f  NASA %.1f  heavy-calls %.0f%%  lat %.4fs", v, rep["rmse"], rep["nasa_score"], rep["heavy_call_percent"], rep["latency_mean_s"])
    if rows:
        wide = pd.concat(rows).pivot(index=["id", "engine", "true_rul_capped", "true_rul_official"], columns="variant", values="pred").reset_index()
        wide.to_csv(out / "predictions_all_variants.csv", index=False)
        flat = pd.DataFrame([{"variant": k, **{kk: vv for kk, vv in r.items() if not isinstance(vv, list)}, "rmse_ci95_lo": r["rmse_ci95_bootstrap"][0],
                              "rmse_ci95_hi": r["rmse_ci95_bootstrap"][1]} for k, r in reports.items()]).drop(columns="rmse_ci95_bootstrap", errors="ignore")
        flat.to_csv(out / "metrics_consolidated.csv", index=False)
    mock = bool(args.backend == "mock")
    summary = {"status": "WIRING TEST ONLY - MOCK BACKEND - NOT RESEARCH RESULTS" if mock else "EXPERIMENTAL RUN",
               "track": "FD001", "eval_samples": int(len(test)), "eval_split": "official test, final window per engine",
               "variant_status": status, "variants": reports,
               "heavy_backend": None if pipeline is None else pipeline.heavy.backend.identity, "narrative": narrative,
               "thresholds": {"tau1": config.tau1, "tau2": config.tau2, "alpha": config.alpha},
               "latency_note": "Latency = measured pipeline time; for cache replays the stored live heavy-path time is added back. "
                               "Live heavy time includes the backend's built-in request spacing (MIN_INTERVAL=1.6 s per call).",
               "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
                               "platform": platform.platform()},
               "files_sha256": {"data/train_FD001.txt": sha256(Path(args.rul).parent / "train_FD001.txt"), "data/test_FD001.txt": sha256(Path(args.rul).parent / "test_FD001.txt"),
                                "data/RUL_FD001.txt": sha256(Path(args.rul)), "base.pt": sha256(Path(args.base)),
                                **({"numeric.pt": sha256(Path(args.numeric))} if args.numeric else {}),
                                **({"sage.pt": sha256(Path(args.checkpoint))} if args.checkpoint else {})}}
    (out / "metrics_consolidated.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "model_config.json").write_text(json.dumps({"small_gate": asdict(config), "numerical_heavy": None if numeric is None else numeric["arch"],
                                                      "seed": config.seed, "base_confidence_examples": base_ck["confidence_examples"]}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train-base"); t.add_argument("--data", required=True); t.add_argument("--output", required=True)
    t.add_argument("--epochs", type=int, default=20); t.add_argument("--confidence-epochs", type=int, default=30)
    t.add_argument("--confidence-limit", type=int); t.add_argument("--numeric-epochs", type=int, default=40)
    e = sub.add_parser("evaluate"); e.add_argument("--data", required=True); e.add_argument("--base", required=True)
    e.add_argument("--numeric"); e.add_argument("--output", required=True); e.add_argument("--rul", default="data/CMAPSS/RUL_FD001.txt")
    e.add_argument("--variants", nargs="*", choices=VARIANTS); e.add_argument("--backend", choices=["openai", "claude", "mock"])
    e.add_argument("--checkpoint"); e.add_argument("--wiring-test", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train-base":
        logging.basicConfig(level=logging.INFO)
        print(json.dumps(train_base(PreparedData.load(a.data), Path(a.output), a.epochs, a.confidence_epochs, a.confidence_limit, a.numeric_epochs), indent=2))
    else:
        s = evaluate(a)
        print(json.dumps({"status": s["status"], "variant_status": s["variant_status"]}, indent=2))


if __name__ == "__main__":
    main()
