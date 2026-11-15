# SAGE — GitHub Final QC Audit
**Repository:** yuvrajrajpurohit01/SAGE (PRIVATE)  
**Local staging:** `E:\Master Thesis\SAGE\SAGE_Release_Staging\sage_clean`  
**Status:** Local fixes applied — **NOT pushed** (pending your approval)  
**Original project:** `E:\Master Thesis\SAGE\SAGE_Journal\` — untouched, preserved

---

## 1. CITATION.cff — corrected
**Issue:** Contained placeholders `url: "https://github.com/your-username/sage"`, `repository-code: "https://github.com/your-username/sage"`, `doi: "TODO"`, `license: "TODO"`.

**Fix applied (local, uncommitted):**
- `url` → `https://github.com/yuvrajrajpurohit01/SAGE`
- `repository-code` → `https://github.com/yuvrajrajpurohit01/SAGE`
- Removed `identifiers.doi` block (was `"TODO"` — not genuinely available)
- Removed `license` field (was `"TODO"` — not genuinely available)

**Remaining:** A real DOI and license value can be added once genuinely available (e.g. after manuscript publication / license selection). Nothing invented.

---

## 2. LICENSE — added
**Issue:** No LICENSE file existed; README said "to be added".

**Fix applied (local, uncommitted):**
- Added `LICENSE` (MIT License, copyright Yuvraj Rajpurohit 2026).

**Basis for choice:**
- Code ownership: Yuvraj Rajpurohit (author listed in CITATION.cff).
- Dependency licensing: `numpy` (BSD), `torch` (BSD-style), `pandas` (BSD) — all permissive, compatible with MIT.
- Optional `live` deps (`anthropic`, `sentence-transformers`, `python-dotenv`) are also permissive.

**Pending your approval:** MIT is a standard template, not invented, but confirm this is the license you want before push. If you prefer Apache-2.0, GPL, or another, say so and I'll swap it.

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
**Result:** Single commit `8a1ea30` (fresh history, no legacy sensitive commits). Secret scan of working tree: **0 hits**. No credentials were ever committed. No revocation/rotation needed.

---

## 6. .gitignore — fixed; sanitized .env.example added
**Issue:** `.env.*` ignore pattern would also ignore a tracked `.env.example`.

**Fix applied (local, uncommitted):**
- `.gitignore`: added `!.env.example` exception so a sanitized template can be tracked.
- Added `.env.example` documenting `SAGE_LLM_BACKEND`, `SAGE_LLM_MODEL`, `SAGE_LLM_BASE_URL`, `SAGE_LLM_API_KEY`, `SAGE_CLAUDE_MODEL` with empty values — no real secrets.
- `.env` remains ignored (never committed).

---

## 7. README commands verified against scripts/dependencies
**Issues found and fixed (local, uncommitted):**
- `Requirements: torch, numpy, pandas, scikit-learn` → `torch, numpy, pandas` (scikit-learn is not a dependency; no `sklearn` imports in any `.py`).
- `python -m pytest tests/  # 37/37 pass` → corrected. `pytest` collects **0** tests (tests are `unittest`-based with `*Tests` classes, not `Test*`). Actual command (37 pass):
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

**Verdict:** Accurate for research claims; no change to results. Optional clarification: note in README that the semantic path exists in code/tests but headline results use numerical D2. Left as-is to avoid altering research claims.

---

## 9. Reproducible clean-install and test workflow — added
**Fix applied (local, uncommitted):**
- `.github/workflows/ci.yml`: GitHub Actions CI on push/PR to `main`; Python 3.11 + 3.12; CPU-only `numpy torch pandas` install from PyTorch wheel index; runs all 37 tests; includes a secret-scan step that fails the build on any finding.

---

## 10. Release tag and notes — pending
**Status:** No release tag exists yet.

**Action needed after all checks pass and you approve:**
- Create annotated tag (e.g. `v1.0.0`) on commit `8a1ea30` (plus the local fixes below once committed).
- GitHub Release "SAGE v1.0.0" with notes summarizing: negative gate finding, 37/37 tests, private repo, no API calls required, data integrity (60 clean engines).

---

## LOCAL CHANGES READY (not committed, not pushed)
The following files are modified/added in the local staging repo and **have not been pushed**:
- `CITATION.cff` — URL fixed, placeholders removed
- `.gitignore` — `!.env.example` exception
- `.env.example` — new, sanitized template
- `README.md` — requirements, test command, license line
- `LICENSE` — new (MIT)
- `.github/workflows/ci.yml` — new CI workflow

**Next step on your approval:** I will commit these locally and push `main` to `https://github.com/yuvrajrajpurohit01/SAGE.git`, then create the release tag. Confirm before I proceed. The repository remains PRIVATE.
