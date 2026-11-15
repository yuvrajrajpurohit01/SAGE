# Claims-Evidence Matrix

Every quantitative claim in the manuscript maps to a real experimental artifact.

## Architecture and Design Claims

| Claim | Evidence | Source File |
|-------|----------|-------------|
| Small model: Conv1D-GRU with Conv1d(features→hidden, k=3) → ReLU → GRU(hidden) → Linear(hidden→horizon) | `sage/forecaster/small_model.py` lines 16-27 | `sage/forecaster/small_model.py` |
| Gate: Fuzzy memberships; Linear(window*hidden, 1); sigmoid | `sage/gate/fuzzy_gate.py` lines 20-34 | `sage/gate/fuzzy_gate.py` |
| Gate accepts small when Qs ≥ τ₁ | `sage/gate/fuzzy_gate.py` line 33 | `sage/gate/fuzzy_gate.py` |
| Heavy path: Patch Transformer (d_model=128, 4 layers, 4 heads, patch=5) | `sage/heavy/numerical_heavy.py` lines 12-25 | `sage/heavy/numerical_heavy.py` |
| Verifier: Linear(3×embedding_dim, 1) + sigmoid | `sage/verify/self_reflection.py` lines 45-51 | `sage/verify/self_reflection.py` |
| Decision: accept heavy if Δ = q_s − q_l ≤ τ₂ | `sage/verify/decision.py` lines 20-21 | `sage/verify/decision.py` |
| Fusion: average = (s+l)/2; confidence-weighted | `sage/verify/decision.py` lines 24-28 | `sage/verify/decision.py` |
| D2 verifier trained on confidence-partition traces | `train_journal.py` Stage 3 | `train_journal.py` |
| Engine-balanced gate loss actually used | `train_gate_b.py` training loop | `train_gate_b.py` |

## Gate Experimental Claims (Offline, 3,124 Validation Windows)

| Claim | Value | Source File |
|-------|-------|-------------|
| Gate A (32 examples): RMSE 13.4296, MAE 9.9991, escalation 0.00% | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 42: RMSE 13.4296, MAE 9.9991, escalation 0.29% | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 43: RMSE 13.4296, MAE 9.9991, escalation 38.32% | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 44: RMSE 13.4296, MAE 9.9991, escalation 86.62% | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 43: accepted RMSE 10.32, rejected RMSE 17.30 | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 43: Pearson −0.486, ECE 0.108 | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate B seed 44: Pearson −0.459, ECE 0.377 | `gate_comparison_report.json` | `runs/gate-retrain/gate_comparison_report.json` |
| Gate confidence range [0.675, 0.799] | `gate_validation_report.json` | `runs/gate-validation-offline/gate_validation_report.json` |

## Clean Test-Set Evaluation Claims (60 Engines, 5 Seeds)

| Claim | Value | Source File |
|-------|-------|-------------|
| small_only: RMSE 15.21 ± 0.63 | `results.json` | `runs/journal-eval/results.json` |
| numerical_heavy: RMSE 13.67 ± 0.51 | `results.json` | `runs/journal-eval/results.json` |
| numerical_sage: RMSE 14.66 ± 0.70 | `results.json` | `runs/journal-eval/results.json` |
| always_average: RMSE 13.48 ± 0.44 | `results.json` | `runs/journal-eval/results.json` |
| sage_weighted: RMSE 14.62 ± 0.70 | `results.json` | `runs/journal-eval/results.json` |
| 60 clean test engines | `clean_test_engines.json` | `runs/gate-final-audit/clean_test_engines.json` |
| 40 leaked test engines | `test_set_exposure_assessment.json` | `runs/gate-final-audit/test_set_exposure_assessment.json` |

## Gate Effectiveness vs Oracle Claims (60 Engines, 5 Seeds)

| Claim | Value | Source File |
|-------|-------|-------------|
| Oracle agreement rate | 37.0% ± 9.7% | `gate_effectiveness_eval.py`, `results.json` |
| False escalation rate | 8.7% ± 7.8% | `gate_effectiveness_eval.py`, `results.json` |
| Missed escalation rate | 37.3% ± 15.1% | `gate_effectiveness_eval.py`, `results.json` |
| Selection regret | +2.55 ± 0.20 RMSE | `gate_effectiveness_eval.py`, `results.json` |
| Oracle RMSE | 11.86 ± 0.51 | `gate_effectiveness_eval.py`, `results.json` |

## Threshold Tuning Claims

| Claim | Value | Source File |
|-------|-------|-------------|
| Tuning on validation does not reliably improve test | `threshold_tuning.json` | `runs/journal-tuning/threshold_tuning.json` |
| Best val thresholds vary by seed | `threshold_tuning.json` | `runs/journal-tuning/threshold_tuning.json` |
| Default thresholds (0.60, 0.05) comparable to tuned | `threshold_tuning.json` | `runs/journal-tuning/threshold_tuning.json` |

## Earlier LLM-Based Exploratory Claims (Separate Configuration)

| Claim | Value | Source File |
|-------|-------|-------------|
| LLM heavy_always: RMSE 67.72, MAE 56.60 | `variant_summary.json` | `runs/sage-minimal-e2e/variant_summary.json` |
| LLM small_only: RMSE 13.77, MAE 10.75 | `variant_summary.json` | `runs/sage-minimal-e2e/variant_summary.json` |
| 20 validation windows, 20 engines, seed 2026 | `experiment_metadata.json` | `runs/sage-minimal-e2e/experiment_metadata.json` |
| 1 new API call, 19 cache hits | `cost_summary.json` | `runs/sage-minimal-e2e/cost_summary.json` |

## Additional Outputs Created

| File | Path |
|------|------|
| Training script | `train_journal.py` |
| Evaluation script | `eval_journal.py` |
| Threshold tuning script | `tune_journal.py` |
| Gate effectiveness evaluation | `gate_effectiveness_eval.py` |
| Metrics verification | `verify_metrics.py` |
| Manuscript | `SAGE_Journal_Manuscript.md` |
| Tables | `paper/tables/tables.md` |
| Figures | `paper/figures/figures.md` |
| Claims-Evidence Matrix | `paper/claims_evidence_matrix.md` |
| Limitations | `paper/limitations.md` |
| Reproducibility Checklist | `paper/reproducibility_checklist.md` |
| Quality Control Report | `paper/final_quality_control_report.md` |
