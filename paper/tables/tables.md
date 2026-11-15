# Table 1: Dataset and Partition Statistics

| Partition | Windows | Engines | Role |
|-----------|---------|---------|------|
| Fit | 9,370 | 60 | Small model + numerical heavy fitting |
| Confidence | 3,237 | 20 | Gate + D2 verifier supervision |
| Validation | 3,124 | 20 | Threshold tuning and development |
| Official test | 100 | 100 | Final evaluation (60 clean, 40 exposed) |
| Clean test (subset) | 60 | 60 | Used for all final reported results |

**Notes:**
- All partitions are engine-disjoint (no engine appears in multiple splits).
- Scaler statistics fitted on fit partition only.
- RUL labels capped at 125 cycles.
- 40 of 100 official test engines were previously exposed in gate-calibration diagnostics and are excluded from final evaluation.
- Source: `data/prepared-fd001/`, `runs/gate-final-audit/test_set_exposure_assessment.json`

# Table 2: Architecture and Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Small model architecture | Conv1d(14→32, k=3) → ReLU → GRU(32) → Linear(32→1) |
| Small model window | 50 cycles × 14 sensors |
| Small model horizon | 1 cycle |
| Small model optimizer | Adam, learning rate 0.001 |
| Small model epochs | 20 |
| Gate architecture | Fuzzy memberships; Linear(1,600→1); sigmoid |
| Gate parameters | 1,665 trainable parameters |
| Gate optimizer | Adam, learning rate 0.001 |
| Gate epochs | 30 |
| Gate loss | Engine-balanced MSE |
| Numerical heavy | Patch Transformer, d_model=128, 4 layers, 4 heads, patch=5 |
| Numerical heavy epochs | 40 |
| D2 verifier | Linear(128, 1) + sigmoid |
| D2 verifier epochs | 30 |
| Alpha (confidence target) | 20.0 |
| Tau1 (gate threshold) | 0.60 |
| Tau2 (verifier threshold) | 0.05 |
| Random seeds | 42, 43, 44, 45, 46 |

**Sources:** `sage/forecaster/small_model.py`, `sage/gate/fuzzy_gate.py`, `sage/heavy/numerical_heavy.py`, `sage/verify/self_reflection.py`, `train_journal.py`

# Table 3: Offline Gate Evaluation (3,124 Validation Windows)

| Gate | Confidence Examples | RMSE | MAE | Escalation Rate | Coverage | Pearson(qₛ, \|e\|) | ECE |
|------|---------------------|------|-----|-----------------|----------|-------------------|-----|
| Gate A (32 ex) | 32 | 13.430 | 10.000 | 0.00% | 100.00% | −0.298 | 0.126 |
| Gate B seed 42 | 3,237 | 13.430 | 10.000 | 0.29% | 99.71% | −0.441 | 0.261 |
| Gate B seed 43 | 3,237 | 13.430 | 10.000 | 38.32% | 61.68% | −0.486 | 0.108 |
| Gate B seed 44 | 3,237 | 13.430 | 10.000 | 86.62% | 13.38% | −0.459 | 0.377 |

**Notes:**
- All gates use the same frozen small model; RMSE/MAE identical.
- Difference is in routing behavior (escalation rate).
- Seed 43 shows best balance: 38.32% escalation, best ECE (0.108).
- Seed 44 is too aggressive (86.62% escalation).
- Qs compressed to [0.675, 0.799]; no confidences below 0.675 or above 0.800.
- Source: `runs/gate-retrain/gate_comparison_report.json`

# Table 4: Clean Test-Set Evaluation (60 Engines, 5 Seeds, Mean ± Pop. Std)

| Variant | RMSE | MAE | NASA Score | Heavy Calls | Escalation | Coverage |
|---------|------|-----|------------|-------------|------------|----------|
| small_only | 15.21 ± 0.63 | 10.61 ± 0.57 | 234 ± 44 | 0% | — | — |
| numerical_heavy | 13.67 ± 0.51 | 10.10 ± 0.53 | 163 ± 22 | 100% | — | — |
| numerical_sage (D1+D2) | 14.66 ± 0.70 | 10.54 ± 0.49 | 207 ± 50 | 25% | 49% | 51% |
| always_average (fusion) | **13.48 ± 0.44** | **9.81 ± 0.42** | **160 ± 21** | 100% | — | — |
| sage_weighted | 14.62 ± 0.70 | 10.48 ± 0.45 | 206 ± 50 | 25% | 49% | 51% |

**Notes:**
- Values show mean ± population std across 5 seeds (42–46).
- Engine-clustered bootstrap 95% CI for RMSE in Table 5.
- Best overall: always_average (fusion).
- Numerical heavy outperforms small model (13.67 vs 15.21).
- Selective gating (numerical_sage) does not consistently improve accuracy.
- Source: `runs/journal-eval/results.json`

# Table 5: Gate Effectiveness vs Oracle (60 Engines, 5 Seeds)

| Metric | Value (mean ± pop. std) | 95% CI across seeds |
|--------|-------------------------|---------------------|
| Oracle agreement rate | 37.0% ± 9.7% | [21.7%, 49.0%] |
| False escalation rate | 8.7% ± 7.8% | [0.0%, 18.2%] |
| Missed escalation rate | 37.3% ± 15.1% | [16.5%, 54.8%] |
| Selection regret (RMSE) | +2.55 ± 0.20 | [2.33, 2.85] |
| Oracle RMSE | 11.86 ± 0.51 | [10.99, 12.28] |

**Definitions:**
- **Oracle agreement**: SAGE's chosen source matches the oracle's better source (small or heavy). Fusion selections are not counted as agreement with either.
- **False escalation**: SAGE selected heavy, but small was better (oracle).
- **Missed escalation**: SAGE selected small, but heavy was better (oracle).
- **Selection regret**: SAGE RMSE − oracle RMSE per engine. Positive = SAGE worse than oracle.
- Source: `gate_effectiveness_eval.py`, `runs/journal-eval/results.json`

# Table 6: Threshold Tuning on Validation (3,124 Windows)

| Seed | Best Val (τ₁, τ₂) | Val RMSE | Test RMSE (tuned) | Test RMSE (default 0.60, 0.05) |
|------|-------------------|----------|-------------------|-------------------------------|
| 42 | (0.15, 0.0) | 12.57 | 15.69 | 15.78 |
| 43 | (0.15, 0.0) | 14.30 | 14.86 | 14.88 |
| 44 | (0.80, 0.0) | 15.65 | 14.65 | 14.76 |
| 45 | (0.40, 0.2) | 12.79 | 14.62 | 14.13 |
| 46 | (0.15, 0.0) | 12.90 | 14.69 | 13.73 |

**Notes:**
- Tuning on validation does not consistently improve test performance.
- Default CoLLM thresholds (0.60, 0.05) perform comparably to tuned thresholds.
- Source: `runs/journal-tuning/threshold_tuning.json`

# Table 7: Earlier LLM-Based Exploratory Results (20 Windows, Separate Configuration)

| Variant | RMSE | MAE | NASA Score | Heavy Calls |
|---------|------|-----|------------|-------------|
| small_only | 13.77 | 10.75 | 60.5 | 0 |
| heavy_always | 67.72 | 56.60 | 120,741 | 20 |
| always_average | 37.43 | 32.93 | 1,429 | 20 |
| sage_average (seed 43) | 17.75 | 11.35 | 261.8 | 5 |

**Notes:**
- LLM-based heavy path (NVIDIA Nemotron Lightning), 20 validation windows, seed 2026.
- Simplified D2 (qₗ = qₛ); fusion branch never reached.
- 1 actual API call; 19 cache hits.
- Reported separately as exploratory configuration; not comparable with numerical results.
- Source: `runs/sage-minimal-e2e/variant_summary.json`
