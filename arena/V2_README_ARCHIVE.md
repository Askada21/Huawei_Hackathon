# TechArena selected solution

Run from `.`:

```sh
python3 arena/run_selected.py
```

This reproduces both temporal validations, earlier-period calibration, selected factor model, full-week refit, and verified submission. It writes a new timestamped run directory and CSV, preserving the delivered solution and previous user files. Compact event arrays and fitting-only interaction caches are reused. The local `.techarena_deps` contains PyTorch and optional implicit ALS; existing system packages supply NumPy/SciPy/scikit-learn/LightGBM. Exact versions and source hashes are in `arena_runs/provenance.json`.

## Selected method

1. Stream CSVs into event arrays: one history per event, all ten candidates together. Keep identifiers as opaque lookup keys. Validate histories and chronological ordering.
2. For each cutoff, fit on earlier events only. Deduplicate historical clicks and labelled fitting clicks by `(user, timestamp, item)`. Build normalized sparse user–item interactions. Obtain a rank-64 spectral basis with 25 power iterations and use 32 non-leading components to initialize user/item factors.
3. Fit a history-aware factor model using an 11-outcome softmax (ten candidates plus no-click). Final selected variant adds a 0.5-weight positive-history ranking auxiliary loss, with sampled unobserved controls, not fabricated impression labels. Regularize embeddings and biases. Use all observed candidates in the supervised loss.
4. Train a small LightGBM combiner on out-of-time predictions from January 25 **strictly before January 26**, along with past-only statistics, history recency and candidate-history similarities. Choose its 493 rounds on separate users inside this earlier period. Freeze it for the January 26 evaluation.
5. Refit the factor model on all seven training days for three epochs, selected from temporal validation. Use all available fitting clicks, including warm-up clicks from histories. Apply the frozen combiner to evaluation. No evaluation-derived statistics or parameter updates.

## Valid results

All main January 26 comparisons use exactly the same 35,138 first-user events / 351,380 rows.

| Model | ROC AUC |
|---|---:|
| Smoothed user/item CTR | 0.546477 |
| Ordered daily-statistics LightGBM | 0.608485 |
| Plain normalized SVD, raw score | 0.677058 |
| Converged spectral initialization + factor model | 0.795603 |
| Factor model + auxiliary past-click objective | 0.797924 |
| Clean earlier-period combiner, base factor model | 0.808259 |
| **Selected auxiliary factor model + clean frozen combiner** | **0.808994** |

Another cutoff, January 25, contains 43,964 first-user events. Converged initialization improves raw factor AUC from 0.715156 to 0.736941. On disjoint held-out users in the strictly earlier calibration period, the clean combiner improves 0.740020 to 0.762453. These values have different populations and must not be compared directly to January 26.

Implicit ALS, cosine-only factors, binary row loss, stronger spectral degree smoothing, and extra SVD iterations did not beat the selected recipe. Simple ensemble weights gave differences too small to justify another final model. All attempted model outcomes/configurations are logged in `arena_runs/experiments.jsonl`; see exclusions below.

## Limits and audit

These are local model-selection results, not official leaderboard scores. January 26 was used for model comparisons and early stopping; it is not an untouched final test. The 150-replicate event/user bootstrap measures sampling uncertainty only, not model-selection bias or temporal shift. Private leaderboard generalization and top-10 placement are unknown. The historical tenth-place score 0.832491 is above our measured local AUC and comes from different evaluation data.

The initial stacking experiment accidentally included earlier-cutoff first events occurring in the later validation period. It was corrected before delivery, tested, and rerun. **Ignore `calibrated_converged` and old `*_calibrated.npy` files**; they remain for audit. The deliverable uses `calibrated_clean` and `*_calibrated_clean.npy` exclusively. Machine-readable exclusions are in `arena_runs/invalid_experiments.json`. Factor model validation itself was not affected. See `arena/AUDIT.md`.

No files were uploaded and no leaderboard submissions were spent. Original scripts, models, SQLite databases and `submission.csv` were preserved.
