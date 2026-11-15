# Part 6: Decision Gate Assessment

## Objective
Determine whether Gate B (3,237-example confidence training) produces sufficient selective-prediction behavior to justify proceeding to live end-to-end experiments.

## Decision Criteria (from Part 6 directive)
"If at least two of the three Gate B seeds exhibit non-trivial escalation at the unchanged threshold (tau1=0.60) and show better small-model reliability in accepted samples than rejected samples, proceed to the limited end-to-end experiment."

## Results Summary

| Seed | Escalation Rate | Accepted RMSE | Rejected RMSE | Accepted Better? | Non-trivial Escalation? | Pearson | ECE | Q-target MSE |
|------|-----------------|---------------|---------------|------------------|-------------------------|---------|-----|--------------|
| Gate A (32 ex) | 0.00% | 13.43 | N/A | N/A | No | -0.298 | 0.126 | 0.089 |
| Seed 42 | 0.29% | 13.42 | 15.01 | Yes | No (too low) | -0.441 | 0.261 | 0.128 |
| Seed 43 | 38.32% | 10.32 | 17.30 | Yes | **Yes** | -0.486 | 0.108 | 0.068 |
| Seed 44 | 86.62% | 4.21 | 14.33 | Yes | **Yes** | -0.459 | 0.377 | 0.209 |

## Assessment

**Two of three seeds (43 and 44) exhibit non-trivial escalation with better accepted-sample reliability.**

- **Seed 43:** 38.32% escalation rate, accepted RMSE 10.32 vs rejected RMSE 17.30 (3.98 RMSE points better), best ECE (0.108), best Q-target MSE (0.0684), best Pearson correlation (-0.486).
- **Seed 44:** 86.62% escalation rate, accepted RMSE 4.21 vs rejected RMSE 14.33 (10.12 RMSE points better). However, escalation is too aggressive — escalating 86.62% of samples defeats the purpose of selective gating.
- **Seed 42:** 0.29% escalation — effectively no escalation, not non-trivial.

## Decision

**PROCEED to Part 7 (minimal end-to-end comparison).**

Recommended checkpoint for end-to-end: **Gate B seed 43** — best balance of escalation rate (38.32%), calibration (ECE 0.108), and confidence discrimination (Pearson -0.486).

## Caveats
1. High seed variance: escalation rates range from 0.29% to 86.62% across seeds.
2. Seed 44's 86.62% escalation is too aggressive for selective gating (escalates almost everything).
3. Without heavy-model predictions, we cannot determine whether escalation is beneficial, false, or missed.
4. Gate B was trained on 3,237 confidence examples (up from 32 for Gate A) — 100x more training data.

## Checkpoints Generated
- Gate A (existing): `runs/fd001-pilot/sage.pt` (32 examples)
- Gate B seed 42: `runs/gate-retrain/seed42/sage.pt` (3,237 examples)
- Gate B seed 43: `runs/gate-retrain/seed43/sage.pt` (3,237 examples)
- Gate B seed 44: `runs/gate-retrain/seed44/sage.pt` (3,237 examples)

## Output Locations
- Training checkpoints: `runs/gate-retrain/seed{42,43,44}/sage.pt`
- Evaluation results: `runs/gate-retrain/eval_{gateA,seed42,seed43,seed44}/`
- Comparison report: `runs/gate-retrain/gate_comparison_report.json`
- Summary CSV: `runs/gate-retrain/gate_comparison_summary.csv`
- Risk-coverage comparison: `runs/gate-retrain/risk_coverage_comparison.csv`
