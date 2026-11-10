# V8: time-conditioned preference interactions

**Local AUC 0.8174909942**, compared with **V5 0.8166685543**, on identical 35,138 first-user events / 351,380 rows after January 26. V8's official score is unknown. The last reported official V3 score remains 0.808665; no upload was performed. The target 0.834673 is not established by these results.

## Implementation

V5 already had additive time-dependent user/item biases. V8 adds a shared rank-four residual transform of the pooled user/history preference vector for each UTC six-hour bucket. This changes user–item interactions, not just score intercepts. Four contexts require 1,024 new coefficients. One side of each residual starts at zero, preserving V5 logits exactly at initialization. Residual-matrix L2 regularization is added; the V5 basis, 32 dimensions, observed-impression pairwise objective, actual same-user negative auxiliary controls, optimizer, learning rate and prediction rule are retained.

The first ablation crossed bucket with weekday/weekend, using eight contexts. It improved raw January 26 AUC to 0.813763 from 0.813080, but declined on January 25 (0.751047 vs 0.761332) and the broader January 24 frame (0.712223 vs 0.724477). Before January 25, fitting contains no weekend contexts. This motivated sharing each bucket transform across day types. It also reduces parameter count, so the respective effects of sharing and reduced complexity are not separately identified. The eight-context version was rejected.

## Selected bucket-only model

| First-user cutoff | V5 raw factor | V8 raw factor |
|---|---:|---:|
| January 26, 35,138 events | 0.813080 | 0.814293 |
| January 25, 43,964 events | 0.761332 | 0.762468 |
| January 24, 47,928 events | 0.724477 | 0.728914 |

Earlier checkpoints use selection-end-day 26 and 25 respectively; later first events cannot select them. Every eligible first-user event is included in the reported complete-frame AUC. These retrospective temporal comparisons are not an untouched future test. No rank or context-count grid was searched: one eight-context test followed by the motivated four-context sharing ablation.

## Complete pipeline

The row combiner fits only the 35,230 first-user OOT events actually occurring on January 25; 8,734 later first events are excluded. Its 325 rounds are selected on earlier held-out users. The event combiner uses the same strictly earlier source and 398 selected rounds. The V5 mixture weights remain fixed: 0.25 raw-factor / 0.75 row-combiner logits, then a half event-mass correction.

| Stage | January 26 AUC |
|---|---:|
| Raw contextual factor | 0.814293 |
| Row combiner | 0.816004 |
| Fixed raw/combiner mixture | 0.817098 |
| **Event-corrected final candidate** | **0.817491** |

The final paired whole-event bootstrap (150 replicates) gives delta **+0.00082244**, interval **[0.00052166, 0.00105209]** versus V5. This measures sampling uncertainty only; repeated model selection and temporal shift are not covered. Global improvement is not uniform across subgroups: unseen-fitting-user AUC declines from 0.713269 to 0.712326 on 595 events, while medium-history users improve from 0.799506 to 0.801116. Do not substitute favorable subgroup scores for global AUC.

The evaluation population is predominantly weekdays, has more unseen users and a longer observation horizon than January 26 validation. Therefore the local gain does not guarantee an official gain. V5 is preserved as a fallback.

## Delivery and reproduction

The selected model is refitted on all 6,868,240 labelled candidate rows for two epochs, selected on the main temporal validation. Earlier row/event combiners remain frozen for evaluation inference. Evaluation labels and evaluation-derived fitting statistics are not used. Event-order and feature-schema checks protect every score combination.

New candidate: `submission_techarena_20261102_v8.csv`. The accompanying verification JSON checks the exact header, 529,390 predictions, all original evaluation rowids exactly once, and finite probabilities in [0,1]. Previous submissions/models remain unchanged.

Reproduce selected training, temporal confirmation and verified full-data output in a fresh directory:

```sh
python3 arena/run_v8.py --broad
```

Configurations, epoch logs, code/model hashes, comparison results and verification paths are in `arena_runs/v8_*` and `v8_selected_solution.json`. Tests cover exact neutral initialization, isolated context changes, finite gradients, legacy history compatibility, objective scaling and observed-negative sampling.
