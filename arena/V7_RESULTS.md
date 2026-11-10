# V7: retain V5 after two different modeling tests

The selected solution remains **V5, local AUC0.8166685543**. Its official score is still unknown; the last user-reported official V3 score is0.808665. No uploads were performed.

## Conditional user-profile inference

We froze the completed V5 item vectors, history/time terms and all global parameters, then fitted small regularized user corrections using actual fitting-period impressions. Each correction contains32 latent coordinates and an engagement intercept. The convex ten-candidate-plus-no-click likelihood uses three damped Newton iterations, latent ridge2 and intercept ridge20. Unknown users receive correction0. This differs from stochastic joint matrix factorization and from freezing an initial item table while training another global model.

Jan26 raw AUC fell from **0.813080 to0.805056**. A fixed equal blend scored **0.811008**, also below the raw incumbent. Short-history AUC also declined in this raw comparison. Reject this tested form: extra fitting of individual profiles did not generalize. This does not prove every possible user-inference method would fail. The prior model had already used fitting labels, so we describe this as conditional refinement, not an independent Bayesian posterior.

## Distinct-user sparse item graph

We next tested a different signal: binary user/item click membership, shrunken positive log-lift relative to independent item frequency, and200 neighbors per item. Unlike raw co-occurrence counts or cosine similarities, this corrects popularity and shrinks low-count coincidences. Repeated histories cannot multiply a user's contribution. We built each graph strictly before its own cutoff, using500-item sparse blocks rather than a dense50,000-square matrix.

The graph's mean-history score alone gave **0.572178 AUC**, so it is a weak standalone ranker. Mean/max/latest-history lift were then added to the existing row combiner, trained on35,230 eligible Jan25 OOT events only. All8,734 later first events were excluded; the source day25 checkpoint was selected before Jan26. We preserved V5's raw factor predictions,0.25 logit mixture and half event-mass correction, using unchanged original48-column features for the frozen event model.

The complete candidate scored **0.8167030664**, versus V5 **0.8166685543** on identical35,138 events/351,380 rows. Delta **+0.0000345122**. Paired whole-event bootstrap150 gives delta interval **[-0.0000216235,0.000120034]**, which contains zero. Earlier held-out combiner AUC was also slightly lower than V5's earlier comparison. Added complexity is not justified by this evidence. No neighbor/shrinkage sweep, full evaluation refit or new CSV was performed.

## Safeguards and limitations

Four model-contract tests pass: negative-event adaptation lowers scores, strong ridge preserves baseline, duplicate clicks cannot multiply graph evidence, and empty histories remain neutral. Explicit event-order checks protect every combination. Existing V5 files remain unchanged. No evaluation inputs were used to fit parameters or aggregate statistics. These results use repeatedly examined model-selection data; bootstrap intervals cover sampling variation only, not model-search or temporal-shift uncertainty. The weekday/weekend and observation-horizon mismatches documented in V4 still apply.

Reproduce in a fresh directory:

```sh
python3 arena/run_v7_checks.py
```

Machine-readable comparisons/configurations and logs are stored under `arena_runs/v7_*`. The recommended submission remains `submission_techarena_20261102_v5.csv`; no new official ROC AUC is claimed.
