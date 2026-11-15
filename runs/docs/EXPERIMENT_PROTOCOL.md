# Experiment protocol

1. Acquire and version actual datasets; document licenses, units, time resolution,
   target definition, original publication times and any missing-value handling.
2. Prepare one run per track/subset/ticker/location. Retain the seed, engine assignments
   or temporal boundaries, scaler statistics and target cap/horizon.
3. Fit the small model on fit only. Select architecture and epochs using a documented
   validation protocol; never train on final test targets.
4. Freeze S and the chosen heavy backend. Collect heavy outputs on the confidence
   partition and learn F/R there. Record any sample budget. For large CMAPSS runs,
   overlapping windows can still correlate within engines: report engine counts as
   well as window counts, and consider engine-balanced sampling in later work.
5. Tune thresholds on validation only; retain an untouched CoLLM-preset-C run for CMAPSS.
   Alpha is unit-dependent and currently set before training. Changing alpha requires
   confidence retraining; there is no hidden alpha sweep on the test set.
6. Run the four implemented comparisons on the same final examples, reusing identical
   heavy forecasts. Preserve all traces. For a true standalone heavy latency measure,
   time NexusPipeline directly with an empty cache and account for all four calls.
7. Evaluate average vs weighted fusion, verification removal and separately trained
   narrative/no-narrative variants. Include independent seeds (e.g. 42, 43, 44), mean,
   dispersion, engine-level or time-block uncertainty intervals, and failure counts.
8. Add external CoLLM/OFA, MoiraiAgent and DCATS baselines with their published protocols
   before claiming superior accuracy or novelty. This package does not stand in for them.
9. Add Nexus calibration only after the uncalibrated four-agent path is established.
   Its guidelines must be derived from historical training folds and accepted only via
   a separate hidden validation fold; never learn guidelines from final test errors.

The command-line evaluation halts on a malformed API response. Do not drop failed
samples and present the remaining set as the complete benchmark. Diagnose and rerun
with a documented protocol revision; retain the previous logs. API retries cover
transport/provider errors through the SDK; JSON/schema violations are explicit failures.

The model defaults are initial implementation choices, not optimized hyperparameters.
The original small-model weights, datasets and live API key were unavailable during
this build. All bundled run evidence is explicitly synthetic and offline.
