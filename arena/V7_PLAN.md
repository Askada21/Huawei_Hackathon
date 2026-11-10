# Fixed-space conditional user adaptation

Hypothesis: early stopping of joint V5 training may leave useful fitting-impression information unused in individual user coefficients. Its known-user AUC exceeds unseen-user AUC, but that alone is not proof of the cause. We test a conditional convex user-profile estimator instead of another stochastic factor model.

Hold V5 item vectors, history encoder, time terms, gates and biases fixed. Using only impressions strictly before each cutoff, fit a 33-dimensional same-user correction (32 latent coordinates and one engagement intercept) to the ten-candidate-plus-no-click likelihood. A diagonal ridge prior shrinks this correction to zero: penalty2 per latent coefficient and20 on the intercept. Use three damped Newton iterations with objective-decreasing backtracking. No catalogue items are labelled negative. No-click events and every observed fitting impression remain included. Histories affect the frozen V5 offsets; repeated history copies are not additional click observations.

This is regularized conditional refinement, not an exact Bayesian posterior: the frozen V5 offsets have already seen the fitting labels. Additional fitting may overfit, so validation is essential. Users without fitting impressions retain delta0. No global or user parameters are fitted using evaluation data.

Expected resource use: under roughly1GB additional arrays, small per-user33x33 systems, single-thread BLAS plus2-thread Torch prediction. Preserve V5 and write only new v7 artifacts. Initial test: exact Jan26 first-user frame, compare raw unadapted vs adapted ranking; if promising, compare complete pipeline and another causal cutoff. A weaker standalone model is retained only if a fixed simple blend has a reliable improvement. No ridge sweep on the same validation set. If neither standalone nor a simple blend improves, reject and preserve V5.

## Second family after rejecting conditional refinement

Build a sparse item graph from binary distinct-user positive interactions strictly before each cutoff. Score edges with positive log(cooccurrence / independence expectation), multiplied by count/(count+5) to shrink rare evidence, exclude the diagonal, and keep200 neighbors per target. This differs from raw pair counts, cosine normalization and implicit ALS. Process500 target items at a time instead of allocating a dense50,000-square matrix. Repeated clicks and repeated history copies cannot multiply a user's cooccurrence contribution.

Evaluate mean/max/latest-history edge strengths, then add these three features to the existing earlier-period row combiner. Keep V5 raw predictions, fixed0.25 logit mixture and fixed half-event correction unchanged. Fit the graph before Jan25/Jan26 respectively; fit the combiner on eligible first events actually occurring Jan25 only. Retain only if the identical-frame end-to-end global AUC improves reliably; otherwise do not generate a new CSV or refit for evaluation. No neighbor/shrinkage grid.
