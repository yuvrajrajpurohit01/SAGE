# Final Quality Control Report

## Summary

This document records all corrections made during the journal-readiness upgrade of SAGE. All corrections are based on actual experiment artifacts; no experiments were re-run after results were recorded.

## Correction 1: D2 Verifier Actually Trained

**Issue:** The independent D2 verifier was implemented as architecture only (`reflection=None` in all checkpoints); qₗ was approximated by qₛ in end-to-end experiments.

**Correction:** The numerical D2 verifier is now trained on confidence-partition latent traces from the Patch Transformer heavy model. All 5 seed checkpoints contain a trained verifier. The D2 confidence-gap rule is empirically evaluated on clean test engines.

**Files modified:** `train_journal.py` (new Stage 3), `eval_journal.py` (evaluation with real q_l)

## Correction 2: Gate B Training Loss Actually Engine-Balanced

**Issue:** `train_gate_b.py` defined `engine_balanced_loss()` but never called it; per-batch MSE was used instead.

**Correction:** The training loop now computes per-engine MSE and averages across engines, giving each engine equal weight regardless of window count. Checkpoint metadata records `engine_balanced_actually_used: True`.

**Files modified:** `train_gate_b.py`

## Correction 3: Clean Test-Set Evaluation

**Issue:** 40 of 100 official test engines were previously exposed in gate-calibration diagnostics.

**Correction:** All final results use only the 60 verified-clean test engines. Leaked engines are documented in `runs/gate-final-audit/test_set_exposure_assessment.json` and `clean_test_engines.json`.

**Files modified:** `eval_journal.py` (filters to clean engines), manuscript Section 4.3

## Correction 4: Five-Seed Evaluation with Bootstrap CIs

**Issue:** Earlier results used only 3 seeds with no engine-clustered confidence intervals.

**Correction:** All variants evaluated on 5 seeds (42-46) with engine-clustered bootstrap 95% CIs (2,000 replicates per seed).

**Files modified:** `eval_journal.py`, `tune_journal.py`, manuscript Tables 4-5

## Correction 5: Threshold Tuning on Validation Only

**Issue:** Thresholds were CoLLM presets (0.60, 0.05) with no tuning.

**Correction:** Grid search over τ₁ ∈ {0.15, 0.30, 0.40, 0.60, 0.80} and τ₂ ∈ {0, 0.05, 0.10, 0.20} on validation data only. Tuned thresholds applied to clean test set.

**Files modified:** `tune_journal.py`, manuscript Table 6

## Correction 6: Numerical Heavy Path (No API Required)

**Issue:** Earlier results depended on LLM API calls (only 1 actual call made, backend changed mid-experiment).

**Correction:** All journal-readiness experiments use the numerical Patch Transformer heavy path. Zero API calls required. Results are fully reproducible without external services.

**Files modified:** `train_journal.py`, `eval_journal.py`, manuscript Section 3.3

## Correction 7: Oracle-Based Gate Effectiveness Analysis

**Issue:** Earlier work did not quantify how often the gate agrees with an oracle selector or measure selection regret.

**Correction:** Added oracle analysis computing agreement rate (37.0%), false escalation (8.7%), missed escalation (37.3%), and selection regret (+2.55 RMSE) across 5 seeds.

**Files modified:** `gate_effectiveness_eval.py`, `verify_metrics.py`, manuscript Section 5.3

## Quantitative Accuracy Verification

| Metric | Reported Value | Source File | Verified |
|--------|---------------|-------------|----------|
| small_only RMSE (clean test) | 15.21 ± 0.63 | `results.json` | Yes |
| numerical_heavy RMSE (clean test) | 13.67 ± 0.51 | `results.json` | Yes |
| numerical_sage RMSE (clean test) | 14.66 ± 0.70 | `results.json` | Yes |
| always_average RMSE (clean test) | 13.48 ± 0.44 | `results.json` | Yes |
| Oracle agreement rate | 37.0% ± 9.7% | `results.json` + `gate_effectiveness_eval.py` | Yes |
| False escalation rate | 8.7% ± 7.8% | `results.json` + `gate_effectiveness_eval.py` | Yes |
| Missed escalation rate | 37.3% ± 15.1% | `results.json` + `gate_effectiveness_eval.py` | Yes |
| Selection regret | +2.55 ± 0.20 | `results.json` + `gate_effectiveness_eval.py` | Yes |
| Clean test engines | 60 | `clean_test_engines.json` | Yes |
| Leaked test engines | 40 | `test_set_exposure_assessment.json` | Yes |
| Gate A ECE | 0.126 | `gate_comparison_report.json` | Yes |
| Gate B seed 43 escalation | 38.32% | `gate_comparison_report.json` | Yes |
| Tests passing | 37/37 | `tests/` | Yes |

## Remaining Limitations

1. **Dataset scope:** NASA C-MAPSS FD001 only
2. **Test-set exposure:** 40/100 engines previously exposed; 60 clean used
3. **Validation-set reuse:** Used for development and threshold tuning
4. **Window-level dependence:** Adjacent windows overlap within engines
5. **Single numerical heavy architecture:** One Patch Transformer configuration
6. **No LLM semantic D2:** Numerical verifier only; semantic path not validated

## Conclusion

The manuscript is ready for journal submission with the following caveats:
- Negative results (selective gating does not improve accuracy) are honestly reported
- Test-set exposure and validation-set reuse are explicitly disclosed
- Seed instability is documented
- All results are reproducible without API calls
- 37/37 software tests pass
- Numerical D2 is distinguished from semantic LLM D2; no claim is made about semantic D2
