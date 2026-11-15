# SAGE: Selective Agentic Gating with Evaluation for Remaining Useful Life Prediction

**Abstract**—Remaining Useful Life (RUL) prediction is a critical task in prognostics and health management. Selective agentic gating proposes routing forecasting between an inexpensive small model and an expensive reasoning path based on learned confidence. This paper presents SAGE (Selective Agentic Gating with Evaluation), a framework implementing D1 fuzzy gating, a numerical Patch Transformer heavy path, and a trained D2 independent verifier with confidence-gap decision/fusion for turbofan engine RUL prediction on NASA CMAPSS FD001. We evaluate on a verified-clean held-out test subset of 60 engines (40 of 100 official test engines were previously exposed in gate-calibration diagnostics and are excluded), using five random seeds (42–46) with engine-clustered 95% bootstrap confidence intervals.

Our central research question is whether a learned gate can intelligently choose between a small forecaster and a heavy forecaster while balancing prediction accuracy and computational utilization. The answer is negative. Across five seeds, the gate agrees with an oracle selector (which chooses the lower-error predictor per engine) on only 37.0% of engines (±9.7%). The gate misses 37.3% of cases where heavy would have been better and falsely escalates 8.7% of cases where small would have been better. Selection regret relative to the oracle is +2.55 RMSE points on average. These failures stem from a fundamental limitation: the gate is trained to estimate small-model error, not the benefit of escalation. Its confidence Qs is compressed to [0.675, 0.799] and poorly calibrated (ECE 0.13–0.38), providing insufficient dynamic range for useful routing. Consequently, selective routing (RMSE 14.66 ± 0.70) does not outperform the small model alone (RMSE 15.21 ± 0.63) in a statistically meaningful way, and is surpassed by simple arithmetic fusion (RMSE 13.48 ± 0.44), which captures complementarity between small and heavy errors without requiring accurate confidence estimation. The D2 verifier is trained and functional but does not rescue the gate's poor discrimination. The principal contribution is therefore a carefully bounded negative result: selective agentic gating requires both a competent heavy model and a well-calibrated, discriminative gate—neither of which is guaranteed by the architecture alone.

**Keywords**—Remaining Useful Life, selective prediction, agentic forecasting, fuzzy gating, confidence estimation, oracle analysis, selection regret, turbofan engine, CMAPSS

## 1. Introduction

Predicting the Remaining Useful Life (RUL) of machinery is essential for prognostics and health management. Accurate RUL estimation enables maintenance scheduling before failure while avoiding unnecessary downtime. Modern forecasting approaches range from small, efficient statistical and deep learning models to large, expensive reasoning pipelines.

The central tension is between accuracy and computational cost. A large model may capture complex degradation patterns, but each prediction incurs computational expense. A small forecaster is cheap but may lack capacity for complex patterns. Selective agentic gating proposes a middle path: use the cheap model when it is confident, and escalate to the expensive model only when the small model is uncertain.

SAGE (Selective Agentic Gating with Evaluation) implements this concept for turbofan engine RUL estimation on the NASA CMAPSS FD001 dataset. The framework consists of:

1. **Small forecaster (S)**: A Conv1D–GRU model producing a prediction ŷₛ and learned features φ(x).
2. **Fuzzy gating agent (D1)**: A learnable fuzzy membership network estimating confidence Qs ∈ [0, 1] and deciding whether to accept S or escalate when Qs < τ₁.
3. **Heavy path (L)**: A Patch Transformer numerical model producing ŷₗ.
4. **Independent verifier (D2)**: A learned model estimating confidence Ql ∈ [0, 1] of the heavy prediction from its latent representation, trained on realized error labels.
5. **Decision and fusion (D2 rule)**: Accept heavy when Qs − Ql ≤ τ₂; otherwise fuse small and heavy predictions (arithmetic mean or confidence-weighted).

### 1.1 Research questions

This study is organized around three questions:

**(RQ1)** Can a learned D1 gate intelligently choose between the small forecaster and the heavy forecaster on individual unseen engines?

**(RQ2)** Does the trained D2 verifier improve the confidence-gap decision, and does selective routing outperform the small model alone?

**(RQ3)** Does selective routing provide a useful accuracy–computation trade-off relative to always invoking the heavy model or always fusing?

### 1.2 Contributions

1. **Complete SAGE implementation**: A working selective agentic gating framework with D1 fuzzy gating, a numerical heavy path, a trained D2 verifier, and confidence-gap decision/fusion.
2. **Oracle-based gate-effectiveness evaluation**: We quantify how often the gate agrees with an oracle selector, its false/missed escalation rates, and selection regret, using five seeds on a verified-clean held-out test set.
3. **Negative result with diagnostic value**: Selective routing does not improve accuracy over the small model alone or over simple fusion. We diagnose the cause: the gate estimates small-model error, not the benefit of escalation, and its confidence is compressed and poorly calibrated.
4. **Reproducible artifacts**: All code, checkpoints (five seeds), prepared data, predictions, and experiment logs are released. No API calls are required; all experiments use a numerical heavy path.

## 2. Related Work

NASA C-MAPSS (Saxena & Goebel, 2008) is the standard benchmark for turbofan engine RUL estimation. Deep learning approaches including LSTM, GRU, and Transformer architectures have been applied to this task. Selective prediction allows models to abstain when uncertain (Geifman & El-Yaniv, 2017), and confidence calibration is critical (Naeini et al., 2015; Guo et al., 2017). Mixture of Experts routes inputs to specialized sub-networks (Jacobs et al., 1991; Shazeer et al., 2017). SAGE can be viewed as conditional computation where routing is based on learned confidence.

CoLLM (Wang et al., 2026) is the closest methodological antecedent, combining a small model, fuzzy gate, large model, and learned self-reflection. Nexus (Das et al., 2026) decomposes forecasting into contextualization, macro/micro outlooks, and synthesis. SAGE adapts these ideas with a numerical heavy path and a trained D2 verifier. Unlike CoLLM, SAGE does not assume access to an LLM's hidden states; the verifier operates on the heavy model's latent representation.

## 3. Problem Formulation and SAGE Design

Let X denote an observed multivariate sensor window with T time steps and F real-valued channels. For FD001, T = 50 and F = 14. The target y is the RUL at the end of the observed window, capped at 125 cycles. SAGE returns a scalar estimate ŷ and records the route taken, confidence scores, and the final prediction source.

### 3.1 Small forecaster

ŷₛ = S(X; θₛ), a Conv1D–GRU network processing 50-cycle windows with 14 sensor features:

Conv1d(14→32, k=3) → ReLU → GRU(32) → Linear(32→1)

### 3.2 Fuzzy D1 gate

Qs = F(φ(x); θg), where membership M = exp(−((features − μ)²) / variance) is flattened and mapped through a linear head + sigmoid. Accept small when Qs ≥ τ₁.

### 3.3 Heavy path

ŷₗ = NumericalLarge(X; θL), a Patch Transformer encoder (patch=5, d_model=128, 4 layers, 4 heads) producing a scalar RUL forecast.

### 3.4 Independent verifier D2

Ql = Verifier(latent; θv), a linear sigmoid model estimating confidence from the heavy model's latent representation, trained on realized error labels q*ₗ = 1 − tanh(|ŷₗ − y| / α).

**Numerical D2 vs. semantic LLM D2.** The full SAGE design (following CoLLM and Nexus) envisions a verifier that reads frozen sentence embeddings of agent-generated justifications. That semantic D2 path is specified but not empirically validated in this study: the sentence-transformers encoder was not available locally, and no research-trained semantic verifier checkpoint was produced. The D2 verifier evaluated here is a **numerical D2**: it is trained on the Patch Transformer's latent traces from confidence-partition engines. Results for numerical D2 must not be generalized to semantic D2.

### 3.5 Confidence-gap decision and fusion

Δ = Qs − Ql:
- Δ ≤ τ₂: accept heavy (prediction = ŷₗ)
- Δ > τ₂: fuse ((ŷₛ + ŷₗ)/2 arithmetic mean, or confidence-weighted)

## 4. Dataset and Experimental Methodology

### 4.1 NASA C-MAPSS FD001

Experiments use the FD001 subset of the NASA C-MAPSS turbofan degradation simulation data. FD001 represents a single operating condition and a single fault mode. The preprocessing retains 14 sensor channels, creates sliding windows of length 50 with stride one, and caps RUL labels at 125 cycles. Input scaling statistics are calculated using the fit partition only.

### 4.2 Data Preparation and Partitions

Engine-disjoint splits:

| Partition | Windows | Engines | Role |
|-----------|---------|---------|------|
| Fit | 9,370 | 60 | Small model + numerical heavy fitting |
| Confidence | 3,237 | 20 | Gate + D2 verifier supervision |
| Validation | 3,124 | 20 | Threshold tuning and development |
| Official test | 100 | 100 | Final evaluation (60 clean, 40 exposed) |

### 4.3 Test-Set Leakage and Clean Evaluation Subset

A post-hoc audit found that 40 of the 100 official FD001 test windows had been included in an earlier gate-calibration diagnostic. These 40 windows (from 40 distinct engines) contributed to confidence targets, error distributions, correlations, ECE, and coverage calculations in prior work. The official test set is therefore **not** an independent final evaluation set.

We identify 60 clean test engines that were never exposed in any diagnostic or training procedure. All final reported results use only these 60 clean engines. The 40 exposed engines are listed in `runs/gate-final-audit/test_set_exposure_assessment.json` and are excluded from all final evaluation.

Within each engine, adjacent sliding windows overlap heavily and are statistically dependent. Results are reported at the window level with engine-clustered bootstrap confidence intervals.

### 4.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Small model window | 50 cycles × 14 sensors |
| Small model hidden | 32 |
| Small model optimizer | Adam, lr 0.001 |
| Small model epochs | 20 |
| Gate optimizer | Adam, lr 0.001 |
| Gate epochs | 30 |
| Gate loss | Engine-balanced MSE |
| Gate batch | Full engine per update |
| Numerical heavy | Patch Transformer, d_model=128, 4 layers, 4 heads, patch=5 |
| Numerical heavy epochs | 40 |
| D2 verifier epochs | 30 |
| Alpha (confidence target) | 20.0 |
| Tau1 (gate threshold) | 0.60 |
| Tau2 (verifier threshold) | 0.05 |
| Random seeds | 42, 43, 44, 45, 46 |

### 4.5 Loss Functions

**Small model**: Scaled MSE on fit partition: L = mean(((ŷₛ − y) / target_std)²)

**D1 gate**: Engine-balanced MSE on confidence partition: L = (1/E) Σₑ mean((Qsᵢ − q*ᵢ)²) for i in engine e. Each engine contributes equally regardless of window count.

**Numerical heavy**: Scaled MSE on fit partition (same as small model).

**D2 verifier**: MSE on confidence-partition latent traces: L = mean((Ql − q*ₗ)²) where q*ₗ = 1 − tanh(|ŷₗ − y| / α).

## 5. Experimental Results

### 5.1 Offline Gate Evaluation (3,124 Validation Windows)

All gates share the same frozen small model, so overall RMSE/MAE are identical. Differences are in routing behavior only.

| Gate | Confidence Examples | RMSE | MAE | Escalation Rate | Coverage | Pearson(qₛ, \|e\|) | ECE |
|------|---------------------|------|-----|-----------------|----------|-------------------|-----|
| Gate A (32 ex) | 32 | 13.430 | 10.000 | 0.00% | 100.00% | −0.298 | 0.126 |
| Gate B seed 42 | 3,237 | 13.430 | 10.000 | 0.29% | 99.71% | −0.441 | 0.261 |
| Gate B seed 43 | 3,237 | 13.430 | 10.000 | 38.32% | 61.68% | −0.486 | 0.108 |
| Gate B seed 44 | 3,237 | 13.430 | 10.000 | 86.62% | 13.38% | −0.459 | 0.377 |

Gate confidence Qs shows moderate negative correlation with absolute error, but Qs is compressed to [0.675, 0.799], never producing confidences below 0.675 or above 0.800. At τ₁ = 0.60, Gate A never escalates; Gate B seed 43 escalates 38.32%; seed 44 escalates 86.62%.

### 5.2 Clean Test-Set Evaluation (60 Engines, 60 Windows)

Final evaluation uses only the 60 verified-clean test engines. Five seeds (42–46) with engine-clustered 95% bootstrap CIs.

| Variant | RMSE | MAE | NASA Score | Heavy Calls | Escalation | Coverage |
|---------|------|-----|------------|-------------|------------|----------|
| small_only | 15.21 ± 0.63 | 10.61 ± 0.57 | 234 ± 44 | 0% | — | — |
| numerical_heavy | 13.67 ± 0.51 | 10.10 ± 0.53 | 163 ± 22 | 100% | — | — |
| numerical_sage (D1+D2) | 14.66 ± 0.70 | 10.54 ± 0.49 | 207 ± 50 | 25% | 49% | 51% |
| always_average (fusion) | **13.48 ± 0.44** | **9.81 ± 0.42** | **160 ± 21** | 100% | — | — |
| sage_weighted | 14.62 ± 0.70 | 10.48 ± 0.45 | 206 ± 50 | 25% | 49% | 51% |

Values show mean ± std across 5 seeds. Engine-clustered bootstrap 95% CI for RMSE shown in Section 5.4.

### 5.3 Gate Effectiveness: Oracle Analysis

To determine whether the gate intelligently selects between small and heavy, we compare each SAGE decision against an oracle that selects, per engine, whichever predictor has lower absolute error.

| Metric | Value (5 seeds, mean ± pop. std) |
|--------|----------------------------------|
| Oracle agreement rate | 37.0% ± 9.7% |
| False escalation rate | 8.7% ± 7.8% |
| Missed escalation rate | 37.3% ± 15.1% |
| Selection regret (SAGE − oracle RMSE) | +2.55 ± 0.20 |
| Oracle RMSE | 11.86 ± 0.51 |

Per-seed agreement rates: seed42=40.0%, seed43=38.3%, seed44=36.7%, seed45=20.0%, seed46=50.0%. The agreement rate never exceeds 50%, indicating the gate performs no better than chance at identifying which predictor will be more accurate.

**False escalation** (gate chose heavy, small was better): 8.7% of engines. These are cases where the gate incurred heavy computation for no accuracy benefit.

**Missed escalation** (gate chose small, heavy would have been better): 37.3% of engines. These are cases where the gate failed to invoke the heavy model when it would have reduced error.

**Selection regret** (+2.55 RMSE points on average): The gap between SAGE's actual error and the oracle's best-possible error. A positive regret means SAGE performs worse than the oracle on most engines.

### 5.4 Key Findings

**Finding 1: Numerical heavy outperforms small model.** The Patch Transformer heavy model (RMSE 13.67) outperforms the Conv1D–GRU small model (RMSE 15.21) on clean test engines.

**Finding 2: Simple fusion is best.** Always averaging small and heavy predictions achieves the best overall accuracy (RMSE 13.48), outperforming both small_only and selective routing. This suggests that the errors of the small and numerical heavy models are complementary, and arithmetic averaging captures this complementarity without requiring a learned gate.

**Finding 3: Selective gating does not improve accuracy.** The numerical_sage variant (RMSE 14.66) is not statistically meaningfully better than small_only (RMSE 15.21), and is worse than fusion (13.48). The gate's escalation decisions are highly unstable across seeds (heavy-call rate 0% to 51.7%).

**Finding 4: D2 verifier is trained and functional.** Unlike earlier work where qₗ was approximated by qₛ, the D2 verifier is now trained on confidence-partition numerical heavy traces. However, the confidence-gap rule (Δ = Qs − Ql ≤ τ₂) does not produce beneficial selective routing because the gate confidence is poorly calibrated (ECE 0.13–0.38) and compressed to a narrow range.

**Finding 5: Threshold tuning does not generalize.** Grid search over τ₁ ∈ {0.15, 0.30, 0.40, 0.60, 0.80} and τ₂ ∈ {0, 0.05, 0.10, 0.20} on validation data selects different thresholds per seed, but applying tuned thresholds on test data does not consistently improve RMSE. The default CoLLM thresholds (0.60, 0.05) perform comparably to tuned thresholds.

### 5.5 Engine-Clustered Confidence Intervals

| Variant | RMSE 95% CI | MAE 95% CI | NASA 95% CI |
|---------|-------------|------------|-------------|
| small_only | [14.62, 16.15] | [10.08, 11.56] | [182.6, 297.7] |
| numerical_heavy | [13.01, 14.42] | [9.39, 10.84] | [127.1, 187.1] |
| numerical_sage | [13.77, 15.69] | [9.82, 11.14] | [163.4, 292.1] |
| always_average | [12.86, 14.13] | [9.21, 10.39] | [124.7, 178.2] |
| sage_weighted | [13.71, 15.66] | [9.82, 11.00] | [160.2, 291.7] |

CI computed across 5 seeds with engine-clustered bootstrap (2,000 replicates per seed).

### 5.6 Threshold Tuning Results

Threshold tuning was performed on the 3,124 validation windows only. The best validation thresholds do not transfer reliably to the clean test set:

| Seed | Best Val (τ₁, τ₂) | Val RMSE | Test RMSE (tuned) | Test RMSE (default 0.60, 0.05) |
|------|-------------------|----------|-------------------|-------------------------------|
| 42 | (0.15, 0.0) | 12.57 | 15.69 | 15.78 |
| 43 | (0.15, 0.0) | 14.30 | 14.86 | 14.88 |
| 44 | (0.80, 0.0) | 15.65 | 14.65 | 14.76 |
| 45 | (0.40, 0.2) | 12.79 | 14.62 | 14.13 |
| 46 | (0.15, 0.0) | 12.90 | 14.69 | 13.73 |

Tuning on validation does not consistently improve test performance, indicating potential overfitting to the validation partition.

## 6. Discussion

### 6.1 Can a learned gate intelligently choose between small and heavy?

No. The primary finding of this study is negative. The gate agrees with an oracle selector on only 37.0% of engines, misses 37.3% of cases where heavy would have been better, and falsely escalates 8.7% of cases where small would have been better. Selection regret relative to the oracle is +2.55 RMSE points on average.

The root cause is architectural: the gate is trained to estimate **small-model quality** (q*ₛ = 1 − tanh(|ŷₛ − y|/α)), not the **benefit of escalation** (how much heavy would improve over small). These are different quantities. A sample where the small forecaster has large error may also be difficult for the heavy model. Conversely, a sample where the small forecaster is accurate may still be improvable by heavy. Because the gate's supervision targets the wrong objective, its confidence Qs does not reliably indicate when escalation will help.

### 6.2 Why estimating small-model error does not estimate escalation benefit

The D1 confidence target is a monotonic transformation of the small model's own absolute error. It answers: "How good is the small model's prediction?" The routing decision needs to answer: "How much better would the heavy model's prediction be?" These questions have different answers whenever small and heavy errors are correlated or when heavy's advantage varies independently of small's error.

In our results, heavy is better than small on ~55% of engines (oracle analysis), but the gate's confidence provides insufficient discrimination to identify those engines. Qs is compressed to [0.675, 0.799], and at τ₁ = 0.60 the acceptance threshold sits below nearly all observed confidence values, so the gate rarely escalates. When it does escalate (seeds 44–46), the decisions are not consistently aligned with where heavy would help.

### 6.3 Why fusion outperforms selective routing

Always averaging small and heavy predictions achieves the best accuracy (RMSE 13.48) because:
1. The small and heavy models make different errors (complementary).
2. The gate cannot reliably identify when heavy will be better than small.
3. Simple averaging captures complementarity without requiring accurate confidence estimation.

This is consistent with error-decomposition analysis: averaging helps when errors are complementary and uncorrelated. Selective routing sacrifices this benefit because the gate cannot selectively retain the cases where averaging helps.

### 6.4 D2 verifier: trained but not sufficient

The D2 verifier was trained on confidence-partition numerical heavy traces and produces calibrated Ql estimates. However, the confidence-gap decision rule depends on **both** Qs and Ql being well-calibrated and discriminative. Since Qs is compressed and poorly calibrated, the gap Δ = Qs − Ql does not provide a reliable routing signal. D2 is functional but does not rescue the gate's poor discrimination.

**Important distinction:** This is numerical D2 (latent traces from a Patch Transformer). The semantic D2 path described in the full SAGE design (frozen sentence embeddings of agent justifications, following CoLLM and Nexus) is specified but not empirically validated. No claim about semantic D2 effectiveness is made or supported by these results.

### 6.5 Comparison with earlier LLM-based exploratory results

Earlier exploratory work used an LLM-based four-agent heavy path (NVIDIA Nemotron) on 20 validation windows. The LLM heavy agent performed substantially worse than the small model (RMSE 67.72 vs 13.77), consistently predicting near the RUL cap (125 cycles). Those results are reported separately as configuration-specific exploratory evidence and should not be compared directly with the numerical results herein.

### 6.6 Limitations

1. **Dataset scope**: All experiments use NASA C-MAPSS FD001 only.
2. **Test-set exposure**: 40 of 100 official test engines were previously exposed; results use only the 60 verified-clean engines.
3. **Validation-set reuse**: The 3,124 validation windows were used for development and threshold tuning.
4. **Window-level dependence**: Adjacent windows within an engine overlap heavily; engine-clustered bootstrap CIs address dependence but do not eliminate it.
5. **Single numerical heavy architecture**: Only one Patch Transformer configuration was tested.
6. **No LLM semantic D2**: The D2 verifier uses numerical latent traces, not frozen sentence embeddings of agent justifications. The semantic D2 path is not empirically validated.

## 7. Conclusion

SAGE implements a complete selective agentic gating framework for RUL prediction with a trained D2 verifier. On NASA CMAPSS FD001, the numerical heavy model (Patch Transformer) outperforms the small Conv1D–GRU forecaster, and simple arithmetic fusion achieves the best overall accuracy. However, selective gating does not consistently improve accuracy over the small model alone, because the gate's confidence is poorly calibrated and unstable across seeds.

The central research question—whether a learned gate can intelligently choose between a small and heavy forecaster while balancing accuracy and computation—is answered negatively for the current design. The D1 gate estimates small-model error, not the benefit of escalation, and its confidence is too compressed and poorly calibrated to support useful routing. The D2 verifier is trainable and functional but cannot compensate for the gate's poor discrimination.

The main contribution is an honest, reproducible evaluation showing that selective agentic gating requires both a competent heavy model **and** a well-calibrated, discriminative gate trained on the correct objective (escalation benefit, not small-model quality). Neither condition is met in the current configuration. Negative results (selective gating does not help) are the principal contribution and should inform future work: future gates should be trained to directly estimate the expected improvement from delegation, and confidence supervision should target the routing decision rather than small-model quality.

## References

[1] H. Wang, L. Ren, T. Zhao, and L. Jiao, "CoLLM: Industrial large–small model collaboration with fuzzy decision-making agent and self-reflection," IEEE Trans. Fuzzy Syst., vol. 34, no. 4, pp. 1120–1130, 2026, doi: 10.1109/TFUZZ.2025.3594229.

[2] S. S. S. Das, P. Goyal, M. Parmar, N. Peng, V. Tirumalashetty, C.-L. Li, R. Zhang, J. Yoon, and T. Pfister, "Nexus: An agentic framework for time series forecasting," arXiv preprint arXiv:2605.14389, May 2026.

[3] A. Saxena and K. Goebel, "Turbofan engine degradation simulation data set," NASA Ames Prognostics Data Repository, 2008.

[4] Y. Geifman and R. El-Yaniv, "SelectiveNet: A deep neural network with an integrated reject option," in Proc. ICML, 2019, pp. 2151–2159.

[5] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. ICML, 2017, pp. 1321–1330.

[6] M. P. Naeini, G. F. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in Proc. AAAI, 2015, pp. 2901–2907.

[7] R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," J. Mach. Learn. Res., vol. 11, pp. 1605–1641, 2010.

## Appendix A. Reproducibility Record

| Item | Location |
|------|----------|
| Training script | `train_journal.py` |
| Evaluation script | `eval_journal.py` |
| Threshold tuning script | `tune_journal.py` |
| Gate effectiveness evaluation | `gate_effectiveness_eval.py` |
| Clean test engines | `runs/gate-final-audit/clean_test_engines.json` |
| Training checkpoints (5 seeds) | `runs/journal-training/seed{42-46}/` |
| Evaluation results | `runs/journal-eval/results.json` |
| Threshold tuning results | `runs/journal-tuning/threshold_tuning.json` |
| Test exposure assessment | `runs/gate-final-audit/test_set_exposure_assessment.json` |

All code, checkpoints, and results are available for reproduction. No API calls were made; all experiments use the numerical heavy path.

## Appendix B. Implementation Verification

37 software tests pass, covering: routing, gating, D2 decision, fusion arithmetic, confidence targets, metrics, data splits, narrative causality, and threshold selection. All tests run without API calls.
