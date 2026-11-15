# Equation mapping and design decisions

Primary sources: user-supplied CoLLM PDF (IEEE TFS 34(4), April 2026,
DOI 10.1109/TFUZZ.2025.3594229), Nexus PDF (arXiv:2605.14389), project brief,
and CDS3007 documentation. The supplied papers control the routing mathematics;
implementation choices that the papers do not specify are identified below.

| Source | Code | Implementation |
|---|---|---|
| CoLLM (1) | `sage/forecaster/small_model.py` | `y_s = S(x; theta_s)`; Conv1D/GRU chosen for SAGE |
| CoLLM (2), (3) | `sage/gate/fuzzy_gate.py` | Feature-based Qs; accept S iff Qs >= tau1 |
| CoLLM (4) | `sage/pipeline.py` | SAGE substitutes an agent pipeline for single L; this is an adaptation |
| CoLLM (5), (12) | `sage/verify/self_reflection.py` | Flatten feature representation; single FC projection + sigmoid |
| CoLLM (6) | `sage/verify/decision.py` | Delta = Qs - Ql; accept heavy iff Delta <= tau2 |
| CoLLM (7) | `sage/verify/decision.py` | On D2=0, arithmetic average of small and heavy outputs |
| CoLLM (8) | `sage/gate/fuzzy_gate.py` | exp(-(z-mu)^2 / sigma^2), with no extra factor of 2 |
| CoLLM (9) | `sage/gate/fuzzy_gate.py` | Retain all timestep/feature membership values in M |
| CoLLM (10) | `sage/gate/fuzzy_gate.py` | 1 - tanh(abs(y_s-y*)/alpha), detached training label |
| CoLLM (11) | `sage/gate/fuzzy_gate.py` | Flatten fuzzy M then affine map + sigmoid |
| CoLLM (13) | `sage/verify/self_reflection.py` | 1 - tanh(abs(y_l-y*)/alpha), detached training label |
| CoLLM (14) | `sage/verify/self_reflection.py` | Sum scalar squared confidence errors / batch size |
| CoLLM (15) | `sage/data/datasets.py`, `sage/inference.py` | (x-mean)/std; fit-only statistics |
| CoLLM (16), (17) | `eval/benchmark.py` | RMSE and MAE |
| Nexus §3.1 | `sage/heavy/nexus_pipeline.py` | Chronological context timeline |
| Nexus §3.2 | same | Independent macro and micro outlooks |
| Nexus §3.3 | same | Synthesis with empty guidelines; calibration deferred |

## Explicit adaptations

- Eq. (11) uses phi notation while §III-B describes fuzzy features. This implementation
  feeds flattened M to the confidence head, consistent with that description and the
  user's required pipeline. It does not claim an undisclosed tensor layout was reproduced.
- The positive Gaussian variance is parameterized as exp(log_variance). This keeps the
  equation unchanged while making positivity intrinsic to the parameterization.
- For scalar RUL, the confidence target is exactly the paper's scalar formula. For
  multi-step outputs, SAGE averages the elementwise confidence labels into one routing
  scalar. This is an explicit extension: mean(1-tanh(abs(error)/alpha)), not
  1-tanh(mean(abs(error))/alpha). No future labels are inference features.
- All target errors and alpha values use original target units. Small-model training
  scales its regression residual internally for numerical stability, then restores units.
- The paper's “whole dataset” normalization wording is not interpreted as permission to
  use held-out test observations. This package deliberately uses fit-only statistics.
  Constant feature standard deviations are replaced with 1 to make normalization defined.
- The independent verifier uses concatenated frozen embeddings of macro justification,
  average micro-step justifications and synthesis justification. Long text is chunked
  using the encoder's tokenizer and pooled. This is SAGE's feature design, not the
  original single-LM latent-state extractor. It does not use Qs or self-reported LLM confidence.
- The offline hash encoder is a lexical test double only. It is blocked for live training.
- The default D2 outcome is heavy acceptance or fusion. There is no additional confidence
  rule that rejects everything or returns only S after D2. Adding that would change the
  specified algorithm. Errors also do not silently trigger such an undocumented rule.
- Confidence-weighted fusion is labeled experimental and is evaluated separately.
- The API model is frozen; there is no GPT-2 adapter training in the SAGE heavy path.
- Context prompts, JSON contracts and trace pooling are newly written SAGE adaptations.
  Original code was unavailable, so reuse of previously built prompts is not claimed.
- Justifications are explicit short forecast explanations. They are not hidden model
  deliberation, verified causal evidence, or a guarantee that the forecast is correct.

## Limitations to test experimentally

A text-only verification representation may miss numeric mistakes with similar prose.
CMAPSS is RUL estimation from sensors, not extrapolation of an observed RUL series;
zero-shot LLM competence in this domain must be measured, not assumed. Synthetic context
adds no independent information beyond the sensors. Weather and stock results require
careful units and publication-time auditing. A prompt forbidding future knowledge cannot
remove an LLM's pretraining contamination: use appropriately dated evaluation periods and
record the chosen model's relevant cutoff, as in Nexus's evaluation logic.
