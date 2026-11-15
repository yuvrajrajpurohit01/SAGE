# SAGE — GitHub Final QC Audit
**Repository:** yuvrajrajpurohit01/SAGE (PRIVATE)  
**Local staging:** `E:\Master Thesis\SAGE\SAGE_Release_Staging\sage_clean`  
**Status:** Committed and pushed to `main` (commit `bf13002` → CI fix pending push at time of this update)  
**Original project:** `E:\Master Thesis\SAGE\SAGE_Journal\` — untouched, preserved

---

## 1. CITATION.cff — corrected
**Issue:** Contained placeholders `url: "https://github.com/your-username/sage"`, `repository-code: "https://github.com/your-username/sage"`, `doi: "TODO"`, `license: "TODO"`.

**Fix applied and pushed:**
- `url` → `https://github.com/yuvrajrajpurohit01/SAGE`
- `repository-code` → `https://github.com/yuvrajrajpurohit01/SAGE`
- Removed `identifiers.doi` block (was `"TODO"` — not genuinely available)
- Removed `license` field (was `"TODO"` — not genuinely available)

**Remaining:** A real DOI and license value can be added once genuinely available (e.g. after manuscript publication / license selection). Nothing invented.

---

## 2. LICENSE — added
**Issue:** No LICENSE file existed; README said "to be added".

**Fix applied and pushed:**
- Added `LICENSE` (MIT License, copyright Yuvraj Rajpurohit 2026).

**Basis for choice:**
- Code ownership: Yuvraj Rajpurohit (author listed in CITATION.cff).
- Dependency licensing: `numpy` (BSD), `torch` (BSD-style), `pandas` (BSD) — all permissive, compatible with MIT.

---

## 3. GitHub repository description and topics
**Issue:** Set via GitHub web UI, not in the git tree — cannot be fixed by a commit.

**Action needed from you (web UI):**
- Repository: https://github.com/yuvrajrajpurohit01/SAGE
- Settings → General → Description: e.g. "Selective Agentic Gating with Evaluation for Remaining Useful Life Prediction (numerical D2, no API calls required)"
- Settings → General → Topics (suggested): `remaining-useful-life`, `selective-prediction`, `agentic-forecasting`, `fuzzy-gating`, `CMAPSS`, `RUL-prediction`, `research`, `negative-result`

---

## 4. data/ and runs/ sensitive-content inspection
**Method:** Regex secret scan across all tracked files under `data/` and `runs/` (API key, secret key, password, token, AWS key, private key, bearer, OpenAI key patterns).

**Result: 0 secret hits.**

Notes:
- `data/` contains public NASA CMAPSS, weather, stock, ETT datasets (sensor readings, provenance) — no credentials, no PII.
- `runs/` contains experiment artifacts (`.pt` checkpoints, metrics JSON, traces, cache). The word "token" appears only as `input_tokens`/`output_tokens` metric fields — not secrets.
- Test labels (RUL values) in `data/CMAPSS/*` and `runs/gate-final-audit/clean_test_engines.json` are legitimate research data, not secrets. The 60-engine clean-test split is documented in README "Data Integrity." No unintended exposure of sensitive labels beyond standard benchmark practice.
- Original experimental records preserved; nothing deleted.

---

## 5. Git history secret verification
**Result:** Fresh history on `main`; no legacy sensitive commits. Secret scan of working tree: **0 hits**. No credentials were ever committed. No revocation/rotation needed.

---

## 6. .gitignore — fixed; sanitized .env.example added
**Issue:** `.env.*` ignore pattern would also ignore a tracked `.env.example`.

**Fix applied and pushed:**
- `.gitignore`: added `!.env.example` exception so a sanitized template can be tracked.
- Added `.env.example` documenting `SAGE_LLM_BACKEND`, `SAGE_LLM_MODEL`, `SAGE_LLM_BASE_URL`, `SAGE_LLM_API_KEY`, `SAGE_CLAUDE_MODEL` with empty values — no real secrets.
- `.env` remains ignored (never committed).

---

## 7. README commands verified against scripts/dependencies
**Issues found and fixed and pushed:**
- `Requirements: torch, numpy, pandas, scikit-learn` → `torch, numpy, pandas` (scikit-learn is not a dependency; no `sklearn` imports in any `.py`).
- `python -m pytest tests/  # 37/37 pass` → corrected. `pytest` collects **0** tests (tests are `unittest`-based with `*Tests` classes, not `Test*`). Actual command (37 pass locally under Python 3.14):
  ```
  set PYTHONPATH=.        # PowerShell: $env:PYTHONPATH="."
  python tests/test_sage.py
  python tests/test_gating_switching.py
  python tests/test_prompt_v2.py
  python tests/test_fetch_data.py
  ```
- Reproduction commands (`main.py prepare`, `train_journal.py`, `eval_journal.py`, `tune_journal.py`, `gate_effectiveness_eval.py`, `generate_docx.py`, `generate_pdf.py`) all match existing scripts — verified against `sage/cli.py` and file listing.
- License line updated from "(to be added)" → "See LICENSE file."

---

## 8. Numerical D2 vs semantic D2 distinction — confirmed
**README statement:** D2 evaluated = numerical latent traces from Patch Transformer (numerical D2); semantic D2 (frozen sentence embeddings of agent justifications) specified but not empirically validated; no claim about semantic D2 effectiveness.

**Code check:** `sage/heavy/numerical_heavy.py` (`NumericalVerifier` on numerical latent) = numerical D2. `sage/verify/self_reflection.py` (`SelfReflectionAgent` + sentence-transformer `TraceEncoder`) = semantic path, implemented and exercised by tests. The journal's headline negative result is based on numerical D2.

**Verdict:** Accurate for research claims; no change to results.

---

## 9. Reproducible clean-install and test workflow — added and pushed
**Fix applied and pushed:**
- `.github/workflows/ci.yml`: GitHub Actions CI on push/PR to `main`; Python 3.11 + 3.12 matrix; **numpy and pandas installed from PyPI**, **torch installed separately from the PyTorch CPU index** (`https://download.pytorch.org/whl/cpu`); runs all 37 tests; includes a secret-scan step that fails the build on any finding.

**Local test verification:** 37/37 tests pass under Python 3.14 (`test_sage.py` 14, `test_gating_switching.py` 7, `test_prompt_v2.py` 12, `test_fetch_data.py` 4). Python 3.11/3.12 are exercised by GitHub Actions; local environment only has Python 3.14.

**GitHub Actions verification:** Could not be confirmed from this environment (`gh` CLI not installed; no authenticated API access). Check the Actions tab at https://github.com/yuvrajrajpurohit01/SAGE/actions to confirm both 3.11 and 3.12 jobs pass.

---

## 10. Release tag and notes — pending until CI passes
**Status:** No release tag exists yet. Per instruction, no tag will be created until CI passes.

**Action needed after CI passes and you approve:**
- Create annotated tag (e.g. `v1.0.0`) on the CI-passing commit.
- GitHub Release "SAGE v1.0.0" with notes summarizing: negative gate finding, 37/37 tests, private repo, no API calls required, data integrity (60 clean engines).

---

## Repository visibility
**Confirmed PRIVATE.** Visibility was **not** changed (per your instruction).

---

## Commit history (pushed)
```
bf13002 Final QC audit fixes          (HEAD, pushed to origin/main)
8a1ea30 Initial release: SAGE clean release
```

**HEAD commit:** `bf130028f4f7d89d64b6dcbb3b418e04605049f2`  
**Remote:** `https://github.com/yuvrajrajpurohit01/SAGE.git`
