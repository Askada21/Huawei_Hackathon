# V16: smooth confidence for sparse user profiles

**Selected local global row ROC AUC: 0.8202805896**, versus V14 **0.8194028105**, on identical 35,138 first-user events / 351,380 candidate rows after January 26. Difference **+0.0008777791**. Paired whole-event bootstrap (150 replicates, seed 42) gives **[+0.0006489599, +0.0011015032]**. Official V16/V14 scores are unknown; last user-reported V3 public AUC remains 0.808665. No upload was performed.

## Implementation

V14 fully switches from its cold-history path to a known profile at the first fitting event. V16 uses a fixed confidence `w = fitting_impressions / (fitting_impressions + 20)`. Twenty impressions represent two complete events. Scale the user vector, bias and user-time term by w; history scale becomes `w * known_scale + (1-w) * cold_scale`. The zero-exposure and masked-profile paths retain w=0. Confidence uses only the existing fitting-exposure buffer. No new fitted parameters or RNG draws; V14's dropout mask stream and all other settings are preserved. This changes both profile strength and history scale, so their separate effects were not isolated. Only shrinkage 20 was tested.

The option `--profile-shrinkage` defaults to zero, preserving V14 and older checkpoints. Positive shrinkage requires the trained cold-history path (`--user-dropout`). The global gates, within-event pairwise objective, auxiliary weight 2.0, profile dropout 0.1, regularization 1, rank-32 factors, rank-four time transforms and initialization are unchanged. V15's 31-leaf combiner is not used; row/event trees retain 7 leaves.

## Temporal evidence

| Raw first-user frame | V14 | V16 |
|---|---:|---:|
| January 26 | 0.8161383882 | 0.8171197512 |
| January 25 | 0.7752369666 | 0.7785878145 |
| January 24, broader horizon | 0.7377589749 | 0.7403785428 |

Selected epochs: 2, 5 and 9 respectively. Earlier checkpoints are selected strictly before the next cutoff; complete-frame scores include every eligible first-user event. All three initialization basis metadata files match V14 exactly.

The row combiner fits only 35,230 actual January 25 OOT events, excluding 8,734 later first events. Its earlier-user-selected round count is 500, the predefined ceiling; this is not claimed converged. The event combiner uses the same strictly earlier source and 471 selected rounds. Preserve 0.25 factor / 0.75 row-combiner logits and half event correction. Fixed mixture AUC is 0.8200185022, corrected final AUC 0.8202805896.

Read-only final exposure diagnostics support the sparse-profile motivation: users with 1–2 fitting events improve by +0.002063 AUC (3,088 events), 3–5 by +0.001563 (6,115), 6–10 by +0.000663 (8,949), and 11+ by +0.000289 (16,391). Completely unseen users are essentially unchanged, with a numerical decline of 0.000039 on 595 events. History-length groups 0–5, 6–15 and 16–20 improve by +0.001717, +0.000965 and +0.000433. These are diagnostics, not independent/causal tests; subgroup AUCs do not average to global AUC and did not select a subgroup-specific prediction rule.

## Checks and delivery

26 relevant model contract tests pass, including exact convex interpolation of known/cold score paths, hidden-profile identity invariance, finite gradients, old history/context behavior, objectives and observed-negative sampling. The runner's syntax/help was checked. A read-only inference audit recomputes 107 events, including 31 cold and 36 one/two-fitting-event profiles: V14 and V16 both exactly match saved probabilities (maximum absolute difference 0.0), and repeat inference is identical.

Refit on all 6,868,240 labelled rows for two epochs; apply frozen earlier-period row/event combiners. New output: **`submission_techarena_20261102_v16.csv`**. Its verification JSON checks exact header, all 529,390 original evaluation rowids exactly once, finite probabilities and [0,1] bounds. Previous submissions remain unchanged. No evaluation labels or evaluation-derived statistics enter fitting; no upload or paid compute.

Reproduce in fresh artifacts:

```sh
python3 arena/run_v16_checks.py --broad --full
```

Full refit is gated on a positive paired bootstrap interval and earlier raw improvements. Constituent commands were executed; the entire fresh runner was not redundantly rerun. Configs, windows, epoch logs, elapsed training times and feature provenance are under `arena_runs/v16_confidence_d32_day{24,25,26}_s42`, `v16_cal_confidence`, `v16_event_confidence`, `v16_confidence_final_d32_day26_s42_full`, and `v16_final_combiner`. Comparisons: `v16_raw*_vs_v14.json`, `v16_final_vs_v14.json`; diagnostics/audit: `v16_exposure_diagnostics.json`, `v16_inference_audit.json`. Selection, model/code hashes and verification: `v16_selected_solution.json`. Experiment records append to `experiments.jsonl`.

These are retrospective model-selection results on repeatedly examined data. Bootstrap measures sampling variation only, not search bias or evaluation time/population shift. Public/private leaderboard generalization and ranking remain unknown. V14 remains a fallback.
