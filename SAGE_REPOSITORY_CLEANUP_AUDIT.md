# SAGE — Repository Cleanup Audit

**Repository:** yuvrajrajpurohit01/SAGE (PRIVATE)  
**Staging dir:** `E:\Master Thesis\SAGE\SAGE_Release_Staging\sage_clean`  
**Commit baseline:** `9bb28f0` (HEAD, pushed to origin/main)  
**Goal:** Lean, professional, publication-quality research repository  
**Constraint:** Do NOT rewrite Git history; do NOT change experimental results, model logic, checkpoints, or manuscript conclusions; preserve all 37 tests; preserve five-seed experiments, checkpoints, clean-test predictions, confidence intervals, gate-effectiveness metrics, claims-evidence documentation; keep four-agent LLM and numerical-heavy implementations separated.

---

## 1. File-by-File Decisions

### Core SAGE Source Code — KEEP (essential)

| File | Decision | Rationale |
|------|----------|-----------|
| `sage/__init__.py` | KEEP | Package marker |
| `sage/config.py` | KEEP | Central hyperparameter Config; used by every module |
| `sage/forecaster/__init__.py` | KEEP | Package marker |
| `sage/forecaster/small_model.py` | KEEP | Conv1D-GRU small forecaster (S, CoLLM eq. 1) |
| `sage/gate/__init__.py` | KEEP | Package marker |
| `sage/gate/fuzzy_gate.py` | KEEP | FuzzyDecisionAgent D1 gate (CoLLM eqs. 2,3,8-11) |
| `sage/heavy/__init__.py` | KEEP | Package marker |
| `sage/heavy/nexus_pipeline.py` | KEEP | **Four-agent LLM heavy path** (Nexus §3.1-3.3). Kept clearly separate from numerical_heavy.py |
| `sage/heavy/numerical_heavy.py` | KEEP | **Numerical Patch Transformer heavy + D2 verifier**. Kept clearly separate from nexus_pipeline.py |
| `sage/inference.py` | KEEP | Single-sample prediction entry point |
| `sage/pipeline.py` | KEEP | SAGEPipeline orchestrator (gate → heavy → verifier → decision/fusion) |
| `sage/trace.py` | KEEP | Verbose trace utility (results-independent, `--verbose` only) |
| `sage/verify/__init__.py` | KEEP | Package marker |
| `sage/verify/decision.py` | KEEP | `decide()` confidence-gap decision/fusion (CoLLM eqs. 6,7) |
| `sage/verify/self_reflection.py` | KEEP | TraceEncoder + SelfReflectionAgent (semantic D2); required by tests |
| `sage/data/__init__.py` | KEEP | Package marker |
| `sage/data/datasets.py` | KEEP | Dataset preparation (CMAPSS/CSV/synthetic); leakage-aware splits |
| `sage/context/__init__.py` | KEEP | Package marker |
| `sage/context/cmapss_narrative.py` | KEEP | Sensor-to-text narrative for FD tracks; required by datasets.py |

### Train Package — KEEP

| File | Decision | Rationale |
|------|----------|-----------|
| `train/__init__.py` | KEEP | Package marker |
| `train/training.py` | KEEP | Progressive training (fit → confidence → gate); `train()` + `load_pipeline()` |

### Eval Package — KEEP (remove `.bak` duplicates)

| File | Decision | Rationale |
|------|----------|-----------|
| `eval/__init__.py` | KEEP | Package marker |
| `eval/benchmark.py` | KEEP | Benchmark evaluation + `metrics()`; variant comparison |
| `eval/benchmark.py.bak5` | **REMOVE** | Obsolete duplicate; no references |
| `eval/benchmark.py.bak6` | **REMOVE** | Obsolete duplicate; no references |
| `eval/day1.py` | **ARCHIVE** | Day-1 pipeline with overlapping variant logic. Not referenced by journal scripts or tests. Move to `archive/` or remove with note |
| `eval/tune.py` | KEEP | Validation-only threshold tuning |
| `eval/tune.py.bak6` | **REMOVE** | Obsolete duplicate; no references |

### Top-Level Scripts — DEDUPLICATE

| File | Decision | Rationale |
|------|----------|-----------|
| `main.py` | **MERGE** | 3-line wrapper: `from sage.cli import main`. pyproject.toml already defines `sage = "sage.cli:main"` console entry point. Remove `main.py`; users run `python -m sage.cli` or installed `sage` command |
| `train_journal.py` | KEEP | Five-seed numerical-D2 training (small + gate + numerical heavy + verifier). Required to reproduce runs/journal-training checkpoints |
| `eval_journal.py` | KEEP | Five-seed numerical-D2 evaluation on clean test engines. Required to reproduce runs/journal-eval results |
| `tune_journal.py` | KEEP | Validation-only threshold tuning for numerical_sage; applies to clean test set |
| `gate_effectiveness_eval.py` | KEEP | Oracle-based gate effectiveness analysis (negative result evidence) |
| `verify_metrics.py` | KEEP | Aggregate metrics computation from journal-eval results.json |
| `fetch_data.py` | KEEP | Weather/stock data fetching utility for CSV tracks |

### Tests — KEEP ALL (37 tests)

| File | Decision | Tests | Rationale |
|------|----------|-------|-----------|
| `tests/__init__.py` | KEEP | — | Package marker |
| `tests/test_sage.py` | KEEP | 14 | Core pipeline, gate, decision, metrics tests |
| `tests/test_gating_switching.py` | KEEP | 7 | Gating/switching verification with MockBackend |
| `tests/test_prompt_v2.py` | KEEP | 12 | Prompt versioning, cache keys, checkpoint consistency |
| `tests/test_fetch_data.py` | KEEP | 4 | Weather/stock data fetch validation |

Total: **37 tests** — all preserved, all functional.

### Documentation & Config — KEEP

| File | Decision | Rationale |
|------|----------|-----------|
| `README.md` | KEEP | Installation + reproduction documentation |
| `CITATION.cff` | KEEP | Citation metadata |
| `LICENSE` | KEEP | MIT License |
| `pyproject.toml` | KEEP | Packaging + console entry point |
| `requirements.txt` | KEEP | `-e .` (editable install) |
| `.gitignore` | KEEP | Excludes `.env`, caches, venv |
| `.env.example` | KEEP | Sanitized template (empty values) |
| `.github/workflows/ci.yml` | KEEP | CI on 3.11/3.12 with secret scan |
| `GITHUB_FINAL_QC.md` | **REMOVE** | Process/QC document from prior step; not research content |

### docs/ — KEEP prompts (v2 loads from disk), remove draft

| File | Decision | Rationale |
|------|----------|-----------|
| `docs/prompts/prompt_v1.json` | KEEP | v1 prompt reference (v1 is embedded in code; kept for parity) |
| `docs/prompts/prompt_v2.json` | KEEP | **Required at runtime** by nexus_pipeline.py `load_prompts("v2")` |
| `docs/prompts/prompt_v2_draft.json` | **REMOVE** | Not referenced by any file (grep confirmed zero references) |

### paper/ — KEEP ALL (manuscript + supporting docs)

| File | Decision | Rationale |
|------|----------|-----------|
| `paper/manuscript.md` | KEEP | Final manuscript (markdown) |
| `paper/SAGE_Manuscript_Revised.docx` | KEEP | Final manuscript (DOCX) |
| `paper/claims_evidence_matrix.md` | KEEP | Claims-to-evidence traceability (required by constraint #6) |
| `paper/final_quality_control_report.md` | KEEP | QC report |
| `paper/limitations.md` | KEEP | Threats to validity |
| `paper/references.bib` | KEEP | References |
| `paper/reproducibility_checklist.md` | KEEP | Reproducibility checklist |
| `paper/tables/tables.md` | KEEP | All tables |
| `paper/figures/figures.md` | KEEP | All figures |

### Data — REMOVE raw, keep prepared

| File | Decision | Rationale |
|------|----------|-----------|
| `data/CMAPSS/*.txt` (train/test/RUL) | **REMOVE** | Raw NASA CMAPSS redistribution unnecessary; provide download + prep instructions in README |
| `data/CMAPSS/provenance.json` | **REMOVE** | Raw dataset provenance; remove with raw data |
| `data/CMAPSS/readme.txt` | **REMOVE** | Raw dataset readme; remove with raw data |
| `data/ett/*` | **REMOVE** | Raw ETT datasets; provide prep instructions |
| `data/stock/*` | **REMOVE** | Raw stock datasets; provide prep instructions |
| `data/weather/*` | **REMOVE** | Raw weather datasets; provide prep instructions |
| `data/VALIDATION.json` | **REMOVE** | Raw validation artifact; remove with raw datasets |
| `data/prepared-fd001/*` | KEEP | **Required** to reproduce 5-seed experiments without re-downloading CMAPSS |
| `data/prepared-ett/*` | **REMOVE** | Prepared non-paper tracks; not needed for core results |
| `data/prepared-stock/*` | **REMOVE** | Prepared non-paper tracks; not needed for core results |
| `data/prepared-weather/*` | **REMOVE** | Prepared non-paper tracks; not needed for core results |

### Runs — KEEP only reproducibility-essential artifacts

| File | Decision | Rationale |
|------|----------|-----------|
| `runs/gate-final-audit/clean_test_engines.json` | KEEP | **Required** by train_journal.py, eval_journal.py, tune_journal.py (60 clean engines) |
| `runs/gate-final-audit/corrected_group_analysis.json` | KEEP | Gate-final-audit evidence |
| `runs/gate-final-audit/final_synthesis.json` | KEEP | Gate-final-audit evidence |
| `runs/gate-final-audit/gate_final_audit.json` | KEEP | Gate-final-audit evidence |
| `runs/gate-final-audit/part6_decision_gate_assessment.md` | KEEP | Audit documentation |
| `runs/gate-final-audit/test_set_exposure_assessment.json` | KEEP | Data integrity evidence (40 exposed engines) |
| `runs/journal-training/seed42-46/small_model.pt` | KEEP | **Required** five-seed checkpoints |
| `runs/journal-training/seed42-46/sage.pt` | KEEP | **Required** five-seed checkpoints (gate) |
| `runs/journal-training/seed42-46/numeric.pt` | KEEP | **Required** five-seed checkpoints (numerical heavy + verifier) |
| `runs/journal-training/training_history_all.json` | KEEP | Training history across seeds |
| `runs/journal-eval/results.json` | KEEP | **Required** final experimental results |
| `runs/journal-eval/predictions_summary.json` | KEEP | Per-seed predictions summary |
| `runs/journal-eval/per_sample_predictions.csv` | KEEP | Clean-test per-sample predictions |
| `runs/journal-tuning/threshold_tuning.json` | KEEP | Threshold tuning results |
| `runs/demo-fd001/*` | **REMOVE** | Demo/smoke artifacts; not research results |
| `runs/fd001-day1/*` | **REMOVE** | Day-1 artifacts; superseded by journal results |
| `runs/fd001-pilot/*` | **REMOVE** | Pilot experiments; not final results |
| `runs/fd001-pilot2/*` | **REMOVE** | Pilot experiments; not final results |
| `runs/gate-calibration-diagnostic/*` | **REMOVE** | Diagnostic; clean_test_engines.json captures the outcome |
| `runs/gate-retrain/*` | **REMOVE** | Retrain experiments; not final results |
| `runs/gate-validation-offline/*` | **REMOVE** | Offline validation; not final results |
| `runs/p-narr/*` | **REMOVE** | Prompt narrative experiments |
| `runs/p-nonarr/*` | **REMOVE** | Prompt non-narrative experiments |
| `runs/p-weather/*` | **REMOVE** | Prompt weather experiments |
| `runs/prompt-pilot-fd001/*` | **REMOVE** | Prompt pilot |
| `runs/sage-minimal-e2e/*` | **REMOVE** | Minimal e2e test |
| `runs/smoke-test/*` | **REMOVE** | Smoke test artifacts |
| `runs/eval/*` | **REMOVE** | Duplicate of eval/ package scripts in runs |
| `runs/docs/*` | **REMOVE** | Documentation duplicates |
| `runs/examples/*` | **REMOVE** | Example synthetic outputs |
| All `runs/*/cache/*.json` | **REMOVE** | Generated API cache files; not research artifacts |

### Duplicate `.bak` files — REMOVE ALL

| File | Decision | Rationale |
|------|----------|-----------|
| `sage/cli.py.bak5` | **REMOVE** | Obsolete backup |
| `sage/data/datasets.py.bak7` | **REMOVE** | Obsolete backup |
| `sage/data/datasets.py.bak8` | **REMOVE** | Obsolete backup |
| `sage/heavy/nexus_pipeline.py.bak2` | **REMOVE** | Obsolete backup |
| `sage/heavy/nexus_pipeline.py.bak3` | **REMOVE** | Obsolete backup |
| `sage/heavy/nexus_pipeline.py.bak4` | **REMOVE** | Obsolete backup |
| `sage/heavy/nexus_pipeline.py.bak5` | **REMOVE** | Obsolete backup |
| `sage/heavy/nexus_pipeline.py.bak9` | **REMOVE** | Obsolete backup |
| `sage/inference.py.bak7` | **REMOVE** | Obsolete backup |
| `sage/inference.py.bak8` | **REMOVE** | Obsolete backup |
| `sage/pipeline.py.bak5` | **REMOVE** | Obsolete backup |
| `train/training.py.bak5` | **REMOVE** | Obsolete backup |

---

## 2. Dependency & Import Impact Analysis

### Removed files with zero references (safe to remove)
- All `*.bak*` files: no imports reference them
- `docs/prompts/prompt_v2_draft.json`: grep confirmed zero references
- `GITHUB_FINAL_QC.md`: process document, no code references
- All raw `data/CMAPSS/*.txt`, `data/ett/*`, `data/stock/*`, `data/weather/*`, `data/VALIDATION.json`: datasets.py reads from paths passed at runtime; removing files doesn't break imports
- All `runs/*/cache/*.json`: cache files loaded only at runtime by NexusPipeline if cache_dir points there; not imported by code
- Removed runs artifacts: train_journal.py/eval_journal.py/tune_journal.py reference only `runs/gate-final-audit/clean_test_engines.json`, `runs/journal-training/`, `runs/journal-eval/`, `runs/journal-tuning/` — all KEPT

### Files kept but referencing removed runs paths
- `train_journal.py` line 20: reads `runs/gate-final-audit/clean_test_engines.json` — KEPT
- `eval_journal.py` line 21: reads `runs/gate-final-audit/clean_test_engines.json` — KEPT
- `tune_journal.py` line 21: reads `runs/gate-final-audit/clean_test_engines.json` — KEPT
- `verify_metrics.py` line 4: reads `runs/journal-eval/results.json` — KEPT

### main.py removal impact
- `pyproject.toml` defines `sage = "sage.cli:main"` console script — entry point unaffected
- `main.py` is only `from sage.cli import main` — removing it is safe
- README reproduction commands use `python main.py prepare` — must be updated to `python -m sage.cli prepare` or `sage prepare`

### Data removal impact
- `sage/data/datasets.py` `prepare_cmapss()` reads `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` from a user-supplied `root` path — removing bundled raw data doesn't break the function; users must supply their own CMAPSS directory
- `data/prepared-fd001/arrays.npz` + `metadata.json` are KEPT so reproduction works out-of-the-box

---

## 3. Proposed Final Directory Tree

```
SAGE/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI: Python 3.11/3.12, 37 tests, secret scan
├── .gitignore
├── .env.example                      # Sanitized LLM backend template
├── CITATION.cff
├── LICENSE                           # MIT
├── README.md                         # Installation + reproduction
├── pyproject.toml                    # Packaging + `sage = "sage.cli:main"`
├── requirements.txt                  # -e .
├── sage/                             # Core SAGE source code
│   ├── __init__.py
│   ├── cli.py                        # CLI entry point (demo, prepare, train, evaluate, predict, tune)
│   ├── config.py                     # Config dataclass + PAPER_THRESHOLDS
│   ├── context/
│   │   ├── __init__.py
│   │   └── cmapss_narrative.py       # Sensor-to-text narrative (FD tracks)
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py               # CMAPSS/CSV/synthetic preparation
│   ├── forecaster/
│   │   ├── __init__.py
│   │   └── small_model.py            # Conv1D-GRU small forecaster
│   ├── gate/
│   │   ├── __init__.py
│   │   └── fuzzy_gate.py             # D1 fuzzy confidence gate
│   ├── heavy/
│   │   ├── __init__.py
│   │   ├── nexus_pipeline.py         # Four-agent LLM heavy path (separate from numerical)
│   │   └── numerical_heavy.py        # Numerical Patch Transformer + D2 verifier
│   ├── inference.py                  # Single-sample prediction
│   ├── pipeline.py                   # SAGEPipeline orchestrator
│   ├── trace.py                      # Verbose trace utility
│   └── verify/
│       ├── __init__.py
│       ├── decision.py               # Confidence-gap decision/fusion
│       └── self_reflection.py        # TraceEncoder + SelfReflectionAgent (semantic D2)
├── train/
│   ├── __init__.py
│   └── training.py                   # Progressive training (fit → confidence → gate)
├── eval/
│   ├── __init__.py
│   ├── benchmark.py                  # Benchmark evaluation + metrics()
│   └── tune.py                       # Validation-only threshold tuning
├── train_journal.py                  # Five-seed numerical-D2 training
├── eval_journal.py                   # Five-seed numerical-D2 evaluation (clean test)
├── tune_journal.py                   # Threshold tuning on validation only
├── gate_effectiveness_eval.py        # Oracle-based gate effectiveness
├── verify_metrics.py                 # Aggregate metrics from results.json
├── fetch_data.py                     # Weather/stock data fetching utility
├── docs/
│   └── prompts/
│       ├── prompt_v1.json
│       └── prompt_v2.json            # Required at runtime for v2 prompts
├── data/
│   └── prepared-fd001/
│       ├── arrays.npz                # Prepared CMAPSS FD001 (required for reproduction)
│       └── metadata.json
├── runs/
│   ├── gate-final-audit/
│   │   ├── clean_test_engines.json   # 60 verified-clean engines (REQUIRED)
│   │   ├── corrected_group_analysis.json
│   │   ├── final_synthesis.json
│   │   ├── gate_final_audit.json
│   │   ├── part6_decision_gate_assessment.md
│   │   └── test_set_exposure_assessment.json
│   ├── journal-training/
│   │   ├── seed42/  seed43/  seed44/  seed45/  seed46/
│   │   │   ├── small_model.pt
│   │   │   ├── sage.pt
│   │   │   └── numeric.pt
│   │   └── training_history_all.json
│   ├── journal-eval/
│   │   ├── results.json
│   │   ├── predictions_summary.json
│   │   └── per_sample_predictions.csv
│   └── journal-tuning/
│       └── threshold_tuning.json
├── paper/
│   ├── manuscript.md
│   ├── SAGE_Manuscript_Revised.docx
│   ├── claims_evidence_matrix.md
│   ├── final_quality_control_report.md
│   ├── limitations.md
│   ├── references.bib
│   ├── reproducibility_checklist.md
│   ├── tables/tables.md
│   └── figures/figures.md
└── tests/
    ├── __init__.py
    ├── test_sage.py          (14 tests)
    ├── test_gating_switching.py (7 tests)
    ├── test_prompt_v2.py     (12 tests)
    └── test_fetch_data.py    (4 tests)
```

---

## 4. Files That Must Remain for Reproducibility

These are the **minimum** set required to reproduce all published results without re-downloading data or re-running experiments:

1. `sage/` (entire package) — source code
2. `train/training.py` — training entry point
3. `eval/benchmark.py`, `eval/tune.py` — evaluation + tuning
4. `train_journal.py` — reproduces 5-seed training checkpoints
5. `eval_journal.py` — reproduces 5-seed clean-test evaluation
6. `tune_journal.py` — reproduces threshold tuning
7. `gate_effectiveness_eval.py` — reproduces gate effectiveness analysis
8. `verify_metrics.py` — reproduces aggregate metrics
9. `data/prepared-fd001/` — prepared data (no CMAPSS download needed)
10. `runs/gate-final-audit/clean_test_engines.json` — 60 clean test engines
11. `runs/journal-training/seed42-46/{small_model.pt,sage.pt,numeric.pt}` — 5-seed checkpoints
12. `runs/journal-eval/{results.json,predictions_summary.json,per_sample_predictions.csv}` — published results
13. `runs/journal-tuning/threshold_tuning.json` — tuned thresholds
14. `paper/` — manuscript + supporting docs
15. `tests/` — all 37 tests
16. `README.md`, `CITATION.cff`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `.github/workflows/ci.yml`

---

## 5. Duplicate Functionality Summary

| Area | Files Involved | Decision |
|------|---------------|----------|
| CLI entry point | `main.py`, `sage/cli.py`, pyproject.toml entry point | Remove `main.py` (3-line wrapper); `sage/cli.py` is the canonical entry |
| Benchmark evaluation | `eval/benchmark.py`, `eval/day1.py` | Keep `benchmark.py` (used by tests and journal scripts); `day1.py` has overlapping variant logic — archive/remove |
| Threshold tuning | `eval/tune.py`, `tune_journal.py` | Keep both: `eval/tune.py` is library function; `tune_journal.py` is journal-specific script |
| Four-agent LLM vs numerical heavy | `sage/heavy/nexus_pipeline.py` vs `sage/heavy/numerical_heavy.py` | Kept clearly separate (per constraint #6) |
| D1 gate vs D2 verifier | `sage/gate/fuzzy_gate.py` vs `sage/verify/self_reflection.py` | Both kept; distinct roles |

---

## 6. Implementation Plan (Separate Branch)

1. Create branch `lean-repository` from `main`
2. Remove all REMOVE-listed files via `git rm`
3. Remove `main.py`, update README commands (`python main.py` → `python -m sage.cli`)
4. Add `ARCHIVE/` note if any files are moved rather than deleted
5. Commit all changes on `lean-repository` branch
6. Run all 37 tests to verify
7. Verify documented commands work
8. Prepare clean release ZIP
9. **Do not push or delete without approval**

---

## 7. Size Impact Estimate

- Raw CMAPSS train/test files (~36 MB) removed from `data/`
- All `runs/*/cache/*.json` files (~5 MB) removed
- All pilot/demo/day1 runs artifacts (~15 MB) removed
- All `.bak` files (~100 KB) removed
- **Estimated reduction: ~55+ MB** (from ~147 MB to ~90 MB), while preserving all research results, checkpoints, and the 37-test suite
