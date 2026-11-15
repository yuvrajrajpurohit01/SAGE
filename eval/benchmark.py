from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sage.data.datasets import PreparedData
from sage.pipeline import SAGEPipeline
from sage.trace import TRACE


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    if truth.shape != prediction.shape or not truth.size or not np.isfinite([truth, prediction]).all():
        raise ValueError("Metrics require nonempty, matching finite arrays")
    error = prediction - truth
    rmse = float(np.sqrt(np.mean(error**2)))  # CoLLM eq. 16
    mae = float(np.mean(np.abs(error)))  # CoLLM eq. 17
    nonzero = np.abs(truth) > 1e-8
    mape = float(100*np.mean(np.abs(error[nonzero]/truth[nonzero]))) if nonzero.any() else None
    return {"rmse": rmse, "mae": mae, "mape_percent_nonzero_targets": mape,
            "mape_excluded_near_zero": int((~nonzero).sum())}


def benchmark(pipeline: SAGEPipeline, data: PreparedData, output: str | Path,
              limit: int | None = None, narrative: bool = True, split: str = "test",
              ablations: bool = False) -> dict:
    if split not in {"test", "validation"} or (limit is not None and limit < 1):
        raise ValueError("Evaluate validation or test with a positive optional limit")
    indices = data.indices(split)
    if limit is not None and limit < len(indices):
        # SEEDED_SUBSET: validation windows of one engine are strongly correlated; sample across the split
        indices = indices[:limit] if split == "test" else np.sort(
            np.random.default_rng(pipeline.config.seed).choice(indices, limit, replace=False))
    if not len(indices):
        raise ValueError("No evaluation samples")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    variants = [("small_only", "small_only", "average"), ("heavy_always", "heavy_always", "average"),
                ("gate_only", "gate_only", "average"), ("sage", "sage", pipeline.config.fusion)]
    if ablations:
        variants.extend([("sage_average", "sage", "average"), ("sage_weighted", "sage", "confidence_weighted")])
    report = {"track": data.config.track, "split": split, "samples": len(indices), "synthetic": data.synthetic,
              "backend": pipeline.heavy.backend.identity, "config": asdict(pipeline.config),
              "status": "SMOKE TEST ONLY" if data.synthetic or pipeline.heavy.backend.identity.startswith("mock") else "EXPERIMENTAL RUN",
              "narrative": narrative, "variants": {},
              # Prompt and experiment metadata
              "prompt_version": pipeline.config.prompt_version,
              "prompt_hash": hashlib.sha256(json.dumps(pipeline.heavy._prompts, sort_keys=True).encode()).hexdigest(),
              "effective_agent_prompts": pipeline.heavy._prompts,
              "effective_task_instruction": pipeline.heavy._task_instruction,
              "backend_identity": pipeline.heavy.backend.identity,
              "dataset": data.config.track,
              "random_seed": pipeline.config.seed,
              "checkpoint_identity": str(Path(output).parent / "sage.pt")}
    original_config = pipeline.config
    try:
        for name, mode, fusion in variants:
            pipeline.config = replace(original_config, fusion=fusion)
            records = []
            with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for _k, i in enumerate(indices, 1):
                    TRACE.begin(name, mode, _k, len(indices), data.ids[i], data.y[i].tolist())
                    result = pipeline(torch.from_numpy(data.x[i:i+1]), data.inputs(int(i), narrative), mode)
                    TRACE.end(name, mode, _k, len(indices), data.ids[i], result, data.y[i].tolist())
                    stream.write(json.dumps({"id": data.ids[i], "truth": data.y[i].tolist(), **asdict(result)}, allow_nan=False)+"\n")
                    records.append(result)
            counts = Counter(r.source for r in records)
            heavy_records = [r for r in records if r.heavy is not None]
            token_totals = {key: sum(getattr(trace, key) for r in heavy_records for trace in
                [r.heavy.context, r.heavy.macro, r.heavy.micro, r.heavy.synthesis]) for key in ["input_tokens", "output_tokens"]}
            report["variants"][name] = {**metrics(data.y[indices], np.asarray([r.prediction for r in records])),
                "small_percent": 100*counts["small"]/len(records), "heavy_final_percent": 100*counts["heavy"]/len(records),
                "fusion_percent": 100*counts["fusion"]/len(records), "heavy_invocation_percent": 100*len(heavy_records)/len(records),
                "logical_agent_calls": 4*len(heavy_records), "cache_hits": sum(r.heavy.cache_hit for r in heavy_records),
                "logical_tokens": token_totals, "measured_seconds_including_cache": sum(r.elapsed_seconds for r in records)}
            TRACE.summary(name, report["variants"][name])
    finally:
        pipeline.config = original_config
    TRACE.final_table(report["variants"])
    report["timing_note"] = "Replay/cache timings are not an end-to-end speedup measurement; heavy_always includes small/gate diagnostic overhead. Tokens are logical totals including cached outputs."
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
