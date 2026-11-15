# Figures

## Figure 1: SAGE Architecture

Text description (configuration in Table 2):
- Small Conv1D–GRU forecaster (S): 50-cycle × 14-sensor window, 32 hidden units
- D1 fuzzy gate: Gaussian membership + linear head + sigmoid, τ₁ = 0.60
- Heavy path: Patch Transformer (d_model=128, 4 layers, 4 heads, patch=5)
- D2 verifier: Linear(128, 1) + sigmoid on latent traces, τ₂ = 0.05
- Decision: accept heavy if Qs − Ql ≤ τ₂, else fuse

## Figure 2: Gate Confidence Distribution (All Seeds)

Data from `runs/gate-retrain/gate_comparison_report.json`:
- Qs range: [0.675, 0.799] across all gates
- Mean Qs: ~0.727, Std: ~0.026 (pooled across seeds)
- Per-seed mean Qs: seed42=0.7613, seed43=0.7202, seed44=0.6312, seed45=0.6314, seed46=0.5511
- Only two calibration bins populated: 0.6–0.7 and 0.7–0.8
- Six bins empty (0.0–0.6 and 0.8–1.0)
- Gate confidence is compressed, limiting routing discrimination

## Figure 3: Oracle Agreement vs Selection Regret by Seed

Data from `gate_effectiveness_eval.py`, `runs/journal-eval/results.json`:
- Seed 42: agreement=40.0%, regret=+2.68, heavy-call=0.0%
- Seed 43: agreement=38.3%, regret=+2.32, heavy-call=0.0%
- Seed 44: agreement=36.7%, regret=+2.87, heavy-call=36.7%
- Seed 45: agreement=20.0%, regret=+2.53, heavy-call=10.0%
- Seed 46: agreement=50.0%, regret=+2.37, heavy-call=51.7%
- Aggregate: agreement=37.0% ± 9.7%, regret=+2.55 ± 0.20

## Figure 4: RMSE Comparison by Variant (Clean Test, 60 Engines)

Data from `runs/journal-eval/results.json`:
- small_only: 15.21 ± 0.63
- numerical_heavy: 13.67 ± 0.51
- numerical_sage: 14.66 ± 0.70
- always_average: 13.48 ± 0.44
- sage_weighted: 14.62 ± 0.70
- oracle: 11.86 ± 0.51
- Best: always_average (fusion)

## Figure 5: Threshold Tuning — Validation vs Test RMSE

Data from `runs/journal-tuning/threshold_tuning.json`:
- Validation RMSE improves with tuning (down to 12.57)
- Test RMSE does not consistently improve (13.73–15.69)
- Default thresholds (0.60, 0.05) comparable to tuned thresholds
- Tuning does not generalize to held-out data

## Figure 6: Error Correlation (Qs vs Absolute Error)

Data from `runs/gate-validation-offline/gate_validation_report.json`:
- Pearson(qₛ, |error|): −0.298 (Gate A), −0.486 (Gate B seed 43)
- Moderate negative correlation: higher confidence associated with lower error
- But Qs range too narrow ([0.675, 0.799]) for reliable thresholding

## Figure 7: Clean vs Exposed Test Engines

Data from `runs/gate-final-audit/test_set_exposure_assessment.json`:
- 40 of 100 official test engines previously exposed (40%)
- 60 clean engines used for final evaluation
- Leaked engines distributed across groups A (15), B (5), C (20)
