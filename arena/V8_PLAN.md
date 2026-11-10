# Contextual interaction experiment

V5 includes time/day-dependent additive user/item biases but its latent preference dot product is shared across all contexts. Test a context-dependent residual transform of the combined user/history vector. Eight contexts are the observed UTC six-hour bucket crossed with weekday/weekend. Each has a rank-four32x32 residual with2,048 total new parameters. This can change which objects suit a user, rather than only shifting individual user/item scores.

The residual starts exactly at zero, so initial logits are identical to V5. Keep V5's spectral basis, dimensions, pairwise loss, actual same-user nonclicked auxiliary controls, learning rate, ridge strength, sampling, scoring and row/event mixture unchanged. Penalize residual-matrix mean square. Use the same Jan26 first-user rows and compare raw AUC to0.813080, then the full pipeline against0.816669 if warranted. Do not select matrix rank via a sweep. Require another causal temporal check before recommending a replacement; preserve V5 and never fit evaluation data.

The rank-four transform is small compared with existing embedding tables; two CPU threads and the existing compact caches suffice. Tests cover exact neutral-logit equality, context isolation and finite gradients. If the raw model or a justified simple combination does not improve reliably, reject without a new submission.
