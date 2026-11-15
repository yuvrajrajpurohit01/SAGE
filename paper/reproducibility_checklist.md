# Reproducibility Checklist

## Code and Artifacts

| Item | Location | Status |
|------|----------|--------|
| Small forecaster (Conv1D-GRU) | `sage/forecaster/small_model.py` | Available |
| Fuzzy D1 gate | `sage/gate/fuzzy_gate.py` | Available |
| Numerical heavy (Patch Transformer) | `sage/heavy/numerical_heavy.py` | Available |
| D2 verifier | `sage/verify/self_reflection.py` | Available |
| D2 decision | `sage/verify/decision.py` | Available |
| Dataset preparation | `sage/data/datasets.py` | Available |
| Journal training script | `train_journal.py` | Available |
| Journal evaluation script | `eval_journal.py` | Available |
| Threshold tuning script | `tune_journal.py` | Available |
| Gate effectiveness evaluation | `gate_effectiveness_eval.py` | Available |
| Metrics verification | `verify_metrics.py` | Available |
| Test suite | `tests/` | 37/37 pass |

## Training Artifacts (5 Seeds)

| Checkpoint | Path | Description |
|------------|------|-------------|
| Seed 42 small+gate | `runs/journal-training/seed42/sage.pt` | Small + D1 gate, engine-balanced loss |
| Seed 42 numerical+verifier | `runs/journal-training/seed42/numeric.pt` | Patch Transformer + D2 verifier |
| Seed 43 small+gate | `runs/journal-training/seed43/sage.pt` | Small + D1 gate, engine-balanced loss |
| Seed 43 numerical+verifier | `runs/journal-training/seed43/numeric.pt` | Patch Transformer + D2 verifier |
| Seed 44 small+gate | `runs/journal-training/seed44/sage.pt` | Small + D1 gate, engine-balanced loss |
| Seed 44 numerical+verifier | `runs/journal-training/seed44/numeric.pt` | Patch Transformer + D2 verifier |
| Seed 45 small+gate | `runs/journal-training/seed45/sage.pt` | Small + D1 gate, engine-balanced loss |
| Seed 45 numerical+verifier | `runs/journal-training/seed45/numeric.pt` | Patch Transformer + D2 verifier |
| Seed 46 small+gate | `runs/journal-training/seed46/sage.pt` | Small + D1 gate, engine-balanced loss |
| Seed 46 numerical+verifier | `runs/journal-training/seed46/numeric.pt` | Patch Transformer + D2 verifier |

## Experiment Outputs

| Output | Path | Description |
|--------|------|-------------|
| Evaluation results | `runs/journal-eval/results.json` | Clean test set, 5 seeds, bootstrap CIs |
| Threshold tuning results | `runs/journal-tuning/threshold_tuning.json` | Validation tuning, test application |
| Training history | `runs/journal-training/training_history_all.json` | All stages, all seeds |
| Clean test engines | `runs/gate-final-audit/clean_test_engines.json` | 60 verified-clean engines |
| Test exposure assessment | `runs/gate-final-audit/test_set_exposure_assessment.json` | 40/100 leaked |

## Reproduction Steps

1. Install dependencies: `pip install torch numpy pandas scikit-learn python-docx reportlab pymupdf`
2. Prepare data: `python main.py prepare --track FD001 --source data/CMAPSS --output data/prepared-fd001`
3. Train all components: `python train_journal.py` (seeds 42-46, CPU)
4. Evaluate on clean test: `python eval_journal.py`
5. Threshold tuning: `python tune_journal.py`
6. Gate effectiveness analysis: `python gate_effectiveness_eval.py`
7. Verify metrics: `python verify_metrics.py`
8. Run tests: `python -m pytest tests/` (37/37 pass)
9. Generate manuscript: `python generate_docx.py` and `python generate_pdf.py`

## Random Seeds

- All training: seeds 42, 43, 44, 45, 46
- Bootstrap: seed 42 per evaluation run
- Threshold: τ1 = 0.60, τ2 = 0.05, alpha = 20.0

## Configuration

| Parameter | Value |
|-----------|-------|
| Window size | 50 |
| Features | 14 |
| Horizon | 1 |
| Hidden | 32 |
| Learning rate | 0.001 |
| Batch size | 256 |
| Small model epochs | 20 |
| Gate epochs | 30 |
| Numerical heavy epochs | 40 |
| D2 verifier epochs | 30 |

## Known Limitations Affecting Reproducibility

1. No API calls required — all experiments use numerical heavy path
2. 40 of 100 official test engines previously exposed; 60 clean engines used
3. Validation-set reuse for threshold tuning
4. Engine-clustered bootstrap CIs (2,000 replicates per seed)
