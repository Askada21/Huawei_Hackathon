# V4 diagnosis: retain V3, no new recommended submission

Official public V3 AUC is **0.808665**, from the user's November 1 screenshot. V3 local AUC remains **0.8151557921**. No additional official scores were requested or inferred, and no files were uploaded.

## Sampling and context shift

| Property | Jan26 local validation | Official evaluation inputs |
|---|---:|---:|
| First-user events | 35,138 | 52,939 |
| Weekend share | 100% | 2.80% |
| Maximum time after cutoff | 23.98 hours | 167.93 hours |
| Median history length | 20 | 15 |
| Empty-history share | 0.56% | 1.26% |
| Users absent from fitting events | 1.69% | 2.74% |
| Median hours since last click | 24.60 | 42.55 |

These differences make the final-day local sample imperfect. They do NOT prove the cause of the official-score gap or provide a conversion from local to official AUC. No rows were cherry-picked or reweighted to raise the reported score. Additional first-event temporal backtests use all eligible users and retain entire events.

The old row/event combiners learned on a weekend-only OOT period. A weekday/day-type effect cannot be estimated from a constant weekday/weekend feature in that fitting sample. The factor model itself used both weekdays and weekends; this is a transfer limitation of the frozen combiners, not evidence that the evaluation file or row identifiers are wrong.

## Controlled experiments

1. **Converged weekday fitting and mixed-period calibration.** Train the unchanged V3 global-gate factor architecture before Jan23 and Jan24, selecting checkpoints strictly inside Jan23 and Jan24 respectively. Jan24 raw AUC improves from the earlier fixed-two-epoch0.637734 to0.710126; its same-day selection AUC is0.718418. The Jan23→Jan24 causal calibration backtest gives0.726446 versus Jan24 raw0.710126. These are different data windows from Jan26 and cannot replace its overall score.
2. Train the same small row combiner on eligible OOT Jan24+Jan25 events, excluding all later first events, then evaluate the identical Jan26 rows. The final mixture/event-correction recipe is held unchanged from V3. **Overall AUC0.8152241533**, V3**0.8151557921**. Delta**+0.0000683611**. A150-replicate paired whole-event bootstrap gives delta interval**[-0.00010681,0.00027444]**. This does not establish a reliable improvement, even before accounting for model-selection uncertainty. Reject as a new recommended submission.
3. **Sparse-vector confidence preservation.** Replace hard per-row unit normalization of spectral vectors with soft normalization using a fitting-only median squared norm, rescaling to the same active-row RMS norm1.5. Keep basis, dimensions, objective, regularization and training budget fixed. Raw Jan26 AUC**0.809845**, versus V3 factor**0.811248**. Reject; do not refit this weaker candidate on full data.

No broad parameter sweep was performed. Increasing model complexity, cherry-picking rows, and targeting hidden evaluation labels were not attempted.

## Inference audit

Loaded the saved V3 model checkpoints and recomputed1,198 local validation and1,670 evaluation events, including empty histories. Maximum absolute probability difference from saved factor predictions was exactly0.0 for both. Existing frame/schema checks enforce48 feature names/order and event alignment. Temporal contracts and history masking tests pass. There is no observed row-order or checkpoint-loading error explaining the score gap.

Prediction marginals alone do not expose hidden evaluation outcomes: median candidate probability is0.03350 locally versus0.03353 in evaluation, and median event probability sum0.61069 versus0.61173. Similar score distributions do not imply equally good rankings across these different populations.

Local errors remain substantial: top-1 accuracy inside clicked events is about39.0%, and an average1.75 nonclicked candidates outrank the clicked candidate. This supports further preference-learning work, but no new representation tested in this round reliably beat V3.

## Delivery and reproduction

Retain `submission_techarena_20261101_v3.csv` and its official public score0.808665. No V4 CSV is recommended. Previous models and submissions are preserved.

Audit: `python3 arena/diagnose_v4.py`

Experiment reproduction: `python3 arena/run_v4_checks.py`

Machine-readable results: `arena_runs/v4_selected_decision.json`, `v4_diagnosis.json`, `v4_inference_audit.json`, `v4_final_comparison.json`. Experiment configurations and epoch logs are inside each `v4_*` directory. Bootstrap intervals describe sampling uncertainty only; temporal shift and model search remain limitations.
