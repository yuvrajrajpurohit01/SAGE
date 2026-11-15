# Limitations and Threats to Validity

## 1. Dataset Limitation

Experiments use only NASA CMAPSS FD001 (turbofan engine degradation simulation data). Results may not generalize to other datasets, machinery types, degradation patterns, or forecasting domains.

## 2. Official Test-Set Exposure

40 of 100 official FD001 test engines (40%) were previously included in the gate-calibration diagnostic (`runs/gate-calibration-diagnostic/`). These test windows contributed to confidence targets, error distributions, correlations, ECE, and coverage calculations. The official test set **cannot** serve as an independent final evaluation. Results use only the 60 verified-clean engines.

## 3. Validation-Set Reuse

The 3,124 validation windows were used for both development (prompt design, architecture decisions, threshold exploration) and evaluation. Results may be optimistic due to this reuse.

## 4. Window-Level Dependence

Adjacent sliding windows within one engine overlap heavily and are statistically dependent. Results are reported at the window level with engine-clustered bootstrap confidence intervals, but dependence is not eliminated.

## 5. Single Numerical Heavy Architecture

Only one Patch Transformer configuration (d_model=128, 4 layers, 4 heads, patch=5) was tested. Other heavy architectures may behave differently.

## 6. No LLM Semantic D2

The D2 verifier uses numerical latent traces from the Patch Transformer, not frozen sentence embeddings of agent justifications. The semantic D2 path described in the full SAGE design (following CoLLM and Nexus) is specified but not empirically validated. No claim about semantic D2 effectiveness is made or supported by these results.

## 7. Seed Instability

Gate B escalation rates vary dramatically across seeds (0% to 51.7% heavy-call rate). This indicates the gate is sensitive to initialization and does not learn robust routing. No single seed can be declared definitively superior.

## 8. Threshold Tuning Does Not Generalize

Grid search over τ₁ ∈ {0.15, 0.30, 0.40, 0.60, 0.80} and τ₂ ∈ {0, 0.05, 0.10, 0.20} on validation data selects different thresholds per seed, but applying tuned thresholds on test data does not consistently improve RMSE.

## 9. Missing Baselines

No direct evaluation of MoiraiAgent, DCATS, OFA, or CoLLM baselines. Comparisons are limited to small_only, numerical_heavy, always_average, numerical_sage, and sage_weighted.

## 10. Negative Finding

The principal result is negative: selective agentic gating does not reliably improve accuracy over the small model alone or over simple fusion. This is an honest, bounded finding and should not be presented as evidence that selective gating never works—only that it does not work with the current gate design (confidence trained on small-model error, not escalation benefit).
