# SAGE: Selective Agentic Gating with Evaluation for Remaining Useful Life Prediction

SAGE implements a selective agentic gating framework for turbofan engine Remaining Useful Life (RUL) prediction on NASA CMAPSS FD001. The framework combines a lightweight Conv1D–GRU forecaster, a fuzzy D1 confidence gate, a numerical Patch Transformer heavy path, and a trained D2 independent verifier with confidence-gap decision/fusion.

## Central Research Question

**Can a learned gate intelligently choose between a small forecaster and a heavy forecaster while balancing prediction accuracy and computational utilization?**

## Principal Finding (Negative Result)

**No.** Across five seeds (42–46) on a verified-clean held-out test set of 60 engines:

- The gate agrees with an oracle selector on only **37.0% ± 9.7%** of engines
- The gate misses **37.3% ± 15.1%** of cases where heavy would have been better
- The gate falsely escalates **8.7% ± 7.8%** of cases where small would have been better
- Selection regret relative to the oracle: **+2.55 ± 0.20 RMSE points**
- Selective routing (RMSE 14.66 ± 0.70) does not outperform small-only (RMSE 15.21 ± 0.63) in a statistically meaningful way
- Simple arithmetic fusion achieves the best accuracy (RMSE 13.48 ± 0.44)

The root cause: the gate is trained to estimate **small-model error**, not the **benefit of escalation**. Confidence Qs is compressed to [0.675, 0.799] and poorly calibrated (ECE 0.13–0.38).

## Repository Contents

```
SAGE/
├── sage/                    # Core SAGE package
│   ├── forecaster/small_model.py      # Conv1D-GRU forecaster
│   ├── gate/fuzzy_gate.py             # D1 fuzzy confidence gate
│   ├── heavy/                         # Numerical Patch Transformer heavy path
│   │   ├── nexus_pipeline.py          # Four-agent pipeline interface
│   │   └── numerical_heavy.py         # Patch Transformer + verifier
│   ├── verify/                        # D2 verifier and decision
│   │   ├── self_reflection.py         # Trace encoder + verifier
│   │   └── decision.py                # Confidence-gap decision + fusion
│   ├── data/datasets.py               # Dataset preparation (fit-only scaling)
│   ├── pipeline.py                    # SAGE pipeline orchestrator
│   ├── cli.py                         # Command-line interface
│   ├── inference.py                   # Single-sample inference
│   └── trace.py                       # Execution tracing
├── train/
│   └── training.py                    # Progressive training (fit → confidence → gate)
├── eval/
│   ├── benchmark.py                   # Benchmark evaluation
│   ├── day1.py                        # Day-1 pipeline (train-base + evaluate)
│   └── tune.py                        # Threshold tuning
├── tests/                             # 37 software tests
├── data/                              # Datasets (CMAPSS, prepared)
├── runs/                              # Experiment artifacts
│   ├── journal-training/              # 5-seed training checkpoints
│   ├── journal-eval/                  # Clean test evaluation results
│   ├── journal-tuning/                # Threshold tuning results
│   └── gate-final-audit/              # Test-set exposure assessment
├── paper/                             # Manuscript and supporting docs
│   ├── SAGE_Journal_Manuscript.md     # Manuscript (markdown)
│   ├── tables/tables.md               # All tables
│   ├── figures/figures.md             # All figures
│   ├── claims_evidence_matrix.md      # Claims-to-evidence traceability
│   ├── reproducibility_checklist.md   # Reproducibility checklist
│   ├── final_quality_control_report.md # QC report
│   └── limitations.md                 # Threats to validity
├── train_journal.py                   # Journal training script (5 seeds)
├── eval_journal.py                    # Journal evaluation script
├── tune_journal.py                    # Threshold tuning script
├── gate_effectiveness_eval.py         # Oracle-based gate effectiveness
├── verify_metrics.py                  # Metrics verification
├── generate_docx.py                   # DOCX generator
├── generate_pdf.py                    # PDF generator
├── main.py                            # CLI entry point
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── CITATION.cff                       # Citation metadata
```

## Installation

```bash
pip install -e .
```

Requirements: torch, numpy, pandas

## Reproduction

All experiments use a numerical heavy path — **no API calls required**.

```bash
# 1. Prepare data
python main.py prepare --track FD001 --source data/CMAPSS --output data/prepared-fd001

# 2. Train all components (5 seeds, CPU)
python train_journal.py

# 3. Evaluate on clean test set
python eval_journal.py

# 4. Threshold tuning (validation only)
python tune_journal.py

# 5. Gate effectiveness analysis (oracle comparison)
python gate_effectiveness_eval.py

# 6. Run test suite (37 tests, numerical path only; no API calls)
set PYTHONPATH=.  # Windows PowerShell: $env:PYTHONPATH="."
python tests/test_sage.py
python tests/test_gating_switching.py
python tests/test_prompt_v2.py
python tests/test_fetch_data.py

# 7. Generate manuscript
python generate_docx.py   # SAGE_Journal_Manuscript.docx
python generate_pdf.py    # SAGE_Journal_Manuscript.pdf
```

## Key Results

| Variant | RMSE | MAE | Heavy Calls |
|---------|------|-----|-------------|
| small_only | 15.21 ± 0.63 | 10.61 ± 0.57 | 0% |
| numerical_heavy | 13.67 ± 0.51 | 10.10 ± 0.53 | 100% |
| numerical_sage (D1+D2) | 14.66 ± 0.70 | 10.54 ± 0.49 | 25% |
| always_average (fusion) | **13.48 ± 0.44** | **9.81 ± 0.42** | 100% |

## Data Integrity

40 of 100 official FD001 test engines were previously exposed in gate-calibration diagnostics. All final results use only the 60 verified-clean engines (`runs/gate-final-audit/clean_test_engines.json`).

## Distinction: Numerical D2 vs Semantic LLM D2

The D2 verifier evaluated here is trained on numerical latent traces from the Patch Transformer (numerical D2). The semantic D2 path described in the full SAGE design (frozen sentence embeddings of agent justifications, following CoLLM and Nexus) is specified but **not empirically validated**. No claim about semantic D2 effectiveness is made.

## License

See LICENSE file.

## Citation

See CITATION.cff.
