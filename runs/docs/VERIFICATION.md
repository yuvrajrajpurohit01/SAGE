# Verification record

Verified on 2026-09-10 in a Linux CPU environment. Exact local package versions
are recorded in `tested_environment.json`; Windows commands are provided but were
not executed on a Windows host.

- 14 unittest cases passed, covering the Gaussian formula, label detachment,
  multi-horizon label reduction, D1/D2 equality, both fusion strategies, fast
  exit without heavy calls, heavy acceptance/fusion, malformed output rejection,
  independent macro/micro inputs, caching, training/checkpoint reload, backend
  mismatch rejection, scaler isolation, engine splits, short-engine padding,
  future-context rejection, causal narratives, label-free inference and
  validation-only threshold selection.
- Offline end-to-end demos completed for FD001-shaped, weather-shaped and
  stock-shaped synthetic inputs, including training and saved comparison logs.
- Command-line demo, threshold tuning, evaluation and unseen-history prediction
  completed using the mock backend.
- Source files compiled successfully.
- Editable package build succeeded using installed dependencies and no index.

The sample JSON files in `examples` are synthetic test outputs. Their numeric
errors do not measure real forecasting performance. Model weights are deliberately
not bundled as pretrained research artifacts; the demo regenerates them.

Not verified here: live Anthropic calls, semantic encoder download/inference,
real CMAPSS/stock/weather benchmark scores, GPU behavior or production deployment.

## Dataset/API update

- Actual NASA CMAPSS archive, Open-Meteo ERA5 daily data, Alpha Vantage IBM
  daily-price demonstration response and the authors' ETTh1 CSV were downloaded.
- All selected real-data inputs loaded successfully: FD001, FD003, weather,
  stocks and ETTh1. See data/VALIDATION.json for dimensions and split counts.
- 18 tests pass after adding provider-error, partial-history, end-of-day
  timestamp and snapshot-overwrite tests.
- Real-data preparation is verified; real-data model training/benchmarking and
  historical multimodal context collection remain outstanding.
