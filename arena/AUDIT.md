# Existing solution audit

Official instructions and starter in the workspace exactly match the supplied Downloads copies (SHA-256 checked). Existing artifacts are preserved.

- `validation_v5.py`: aggregates before January 25, fits only a 60% sample of January 25, tests a 50% sample of all January 26 events. This does not approximate the official first-event-per-user sampling. It wastes most labelled rows for supervised fitting.
- `final_submission.py`: reuses the validation model and the January 25 historical-statistics cutoff; does not refit on the complete training week.
- `build_item_similarity.py`: counts overlapping histories repeatedly per event, uses raw pair counts rather than normalized similarity, and truncates to 30 neighbors. This conflates popularity/activity with preferences and loses much of the interaction structure.
- The file named `build_similarity_v7.py` currently contains a validation/training program, not a standalone normalized-similarity builder. Filenames and printed versions cannot establish artifact provenance.
- Changing the historical cutoff and fit window across v5–v8 means their local AUC changes are not controlled comparisons.
- None of these observations alone proves label leakage in v5. The new pipeline rebuilds its own interactions from fitting events only, rather than trusting old SQLite provenance.
- Identifiers remain opaque keys. No attempt is made to infer information from their spelling or recover evaluation labels.

# New validation contract

For cutoff day D, fit on all events strictly before D. Validate on each user's first event on or after D within the available training period. Later events are excluded from validation and from all fitting. All ten rows remain together. Users with no fitting events are retained. Histories on validation events are used only as prediction-time inputs, not as embedding training observations. Repeated historical clicks are deduplicated by (user, time, item).

Integer vocabulary allocation includes evaluation keys only to make encoding stable. Their existence/order is not a feature; there are no evaluation-derived aggregates or parameter updates. Unknown/untrained entities require neutral fallbacks.

## Stacking issue found and corrected before delivery

The first version of the calibrator used all first events after January 25. Some of those occurred on January 26 and overlapped the later validation sample. Those stacking results are INVALID, even though the underlying factor models were fitted correctly. The corrected calibrator filters its training events strictly before January 26 and asserts that boundary. A regression test covers a user whose first event after the earlier cutoff crosses into the later period. `calibrated_converged` and its old frozen-calibration results are retained for audit only; the deliverable uses `calibrated_clean` exclusively. See `invalid_experiments.json` for machine-readable exclusions.
