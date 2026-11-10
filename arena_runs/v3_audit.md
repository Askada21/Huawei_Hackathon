# V3 validation and inference audit

This read-only review examined the original clean baseline and the new `stable_v3.py`, `history_model_v3.py`, `event_component_v3.py`, and `v3_event_apply.py`. It did not train models or modify their source files.

## Baseline verification

The saved selected baseline reproduces ROC AUC **0.8089944032037183** on exactly 35,138 first-user events / 351,380 candidate rows. Its prediction and ordered-event hashes, subgroup metrics and event diagnostics are recorded in `v3_baseline_verification.json`. Known invalid stacking artifacts are explicitly rejected by the new `arena/compare_v3.py` comparison tool.

## Checkpoint-selection nuance in the previous recipe

The day-25 factor checkpoint was selected against the complete first-event-after-January-25 sample. That sample includes 8,734 events occurring on January 26. Removing these events before fitting the LightGBM combiner prevents direct later-label fitting, but does not undo their earlier influence on checkpoint selection.

This is **model-selection contamination, not gradient or aggregate-statistic leakage**. The day-25 embedding parameters were fitted on earlier fitting rows only. Nevertheless, January-26 labels indirectly influenced which checkpoint produced the earlier calibration features. Therefore the previous January-26 AUC must remain a model-selection result, not an independent test result. The existing documentation already makes the broader model-selection limitation explicit.

For new chronological stacking, choose the earlier factor checkpoint using only events strictly before the next evaluation cutoff, or predeclare a fixed epoch count. Then fit the combiner on strictly earlier out-of-time events and freeze the complete recipe for its next temporal comparison. Do not describe a retrospective backtest as an untouched test if its labels or descendants already influenced model selection.

## New factor/history implementations

- `selection-end-day` in both new training scripts correctly restricts the AUC controlling checkpoint selection while retaining all first-user rows for export. Later rows are printed as diagnostics only. Only `selection_auc` affects the selected checkpoint and early stopping.
- Stable models use unchanged `Factors` parameter keys and are compatible with the existing factor export constructor.
- History models restore the correct `HistoryFactors` class and its configuration, including separate history representations and the fitting-only `user_exposure` buffer.
- The history export recreates the same 48 ordered feature names and row flattening convention as the existing clean combiner. Its history geometry columns intentionally continue to use candidate-item geometry, even when the history encoder has separate context embeddings. That preserves the old feature contract; it is not a newly trained cross-embedding similarity feature.
- Recency weights exclude padding, empty histories return zero history contribution, and candidate-specific weights remain per-event. The user-reliability gate uses fitting-only exposure counts. No new future-label fitting path was found.

## Event decomposition and application

The event model uses permutation-invariant summaries of each event's ten candidate rows. Its fitting filter is strictly before January 26 and rejects overlap with supplied test events. Conditional normalization and the fixed correction formula are algebraically consistent between training diagnostics and application:

`0.5 * row_probability + 0.5 * event_probability * row_probability / event_row_probability_sum`

The applied row values remain finite and in [0,1]. Their event sum need not equal one, as allowed by the official submission contract. Event-probability-sum AUC is diagnostic only.

The event component's earlier user split is useful out-of-user selection evidence. It is not a wholly independent confirmation: the event stopping round is chosen on its held-out users, and the pre-existing 493-round row-combiner setting was itself selected using the same earlier split. The inherited day-25 checkpoint-selection nuance also applies whenever the old cache supplies training features.

### Robustness gaps reported to the root agent

1. `event_component_v3.py` accepts `--test-model`, but its clean row-probability source is hardcoded to the selected day-26 joint model. It must verify that the supplied test cache's ordered `events` array equals the clean model's ordered `valid_events.npy` before combining them. The intended current paths are aligned, but a different same-length model could silently misalign.
2. `v3_event_apply.py` verifies row count and feature names but not event identity/order between the feature cache and the supplied row-probability file. Add an explicit events array or provenance check, or constrain callers to known aligned artifacts and document the guarantee. Length alone is insufficient.
3. Application output paths should end in `.npy`: `np.save` automatically appends that suffix otherwise, while the subsequent hash currently refers to the unextended path. Existing intended `.npy` calls are unaffected.

## Comparison discipline

The reported robustness gaps above were subsequently fixed: explicit event-order checks, a required row-events input, and an exact `.npy` suffix guard now reject misaligned or ambiguous calls. The final event model is `v3_event_global_strict`, trained from the NEW global-gate day-25 cache with checkpoint selection strictly before January 26. The old event model and provisional comparisons remain audit artifacts only. See `v3_selected_solution.json` for the authoritative delivered paths.

Use `arena/compare_v3.py` on identical ordered event arrays; all frequency and unseen-user subgroup assignments derive only from fitting rows. Whole-event bootstrap resampling preserves the ten dependent rows and corresponds to resampling distinct validation users. Its confidence interval measures sampling variation only, not model-search bias or temporal distribution shift. Subgroup AUCs need not average to the global AUC and do not establish causal benefits.
