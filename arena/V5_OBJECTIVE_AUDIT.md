# V5 controlled preference-learning experiments

## Existing model and change in supervision

V3 uses an 11-outcome event softmax (ten shown candidates and a no-click outcome). It already learns competition between shown alternatives. Its historical-click auxiliary loss contrasts deduplicated positive interactions against unobserved catalogue controls; these controls were a weak implicit-preference assumption, not observed clicked=0 labels. We do not describe V3 as lacking candidate comparisons.

The first experiment replaces only the supervised cross entropy with a sum of ten winner-versus-loser logistic comparisons per event. The no-click outcome has fixed score0. For a no-click event, it wins against all ten observed candidates, so those events still produce gradients. For a clicked event, the clicked object wins against nine shown nonclicked candidates and no-click. At zero logits, multiplying the mean pair loss by20/11 matches every trainable logit gradient of cross entropy. This controls the initial loss-gradient scale rather than quietly changing effective step size and regularization.

Architecture,32-dimensional basis, spectral initialization, global history/user gates, optimizer, learning rate, L2 strength, sample order and auxiliary coefficient are retained. No evaluation-derived representations or aggregates are fitted. Six objective tests cover matching gradients, correct rankings, no-click gradients, permutation invariance, cross-event observed negatives and an empty-positive edge case.

## Predeclared experiment criteria

- **Event pairwise loss:** retain only with a global row-AUC improvement on identical Jan26 events and confirmation on another cutoff. Within-event top1 alone is insufficient.
- **Hybrid losses:** test one equal mixture only after the individual pairwise objective shows a positive result. Compare CE+within-event pairwise and a CE+cross-event log-probability ranking variant. Cross-event controls are actual clicked=0 rows in the batch. No broad loss-weight sweep.
- **Sparse-user shared estimator:** train a separate history-context table under the original softmax, with L2 and shrinkage toward its initial value, to estimate preferences without relying only on individual user parameters. Compare against the existing frozen-context and shared-table baselines. Reject if overall AUC declines.
- **Observed auxiliary controls:** after a promising pairwise result, replace unobserved catalogue controls with actual fitting-period nonclicked impressions of the same user. Sampling preserves impression multiplicity. Users with no such observations are skipped for this auxiliary term, and same-item comparisons are removed. Positive histories remain deduplicated; repeated history copies are not independently labelled observations. A regression test ensures the pool excludes positive and future rows.

Every new run refuses an existing output directory. Earlier day25 and day24 checkpoints are selected strictly before the next cutoff. Later first-user events remain available for complete-frame diagnostics but cannot select these checkpoints. Jan26 remains model-selection evidence, not an untouched final test. Local scores are never presented as official scores; the last known official V3 score is0.808665.
