# V14: cold-user history path

**Selected local global row ROC AUC: 0.8194028105**, versus V13 **0.8186888601**, on identical 35,138 first-user events / 351,380 rows after January 26. Delta **+0.0007139504**. Paired whole-event bootstrap, 150 replicates, seed 42: **[+0.0005145023, +0.0009356859]**. Official V14/V13 scores are unknown; last user-reported V3 public AUC remains 0.808665. No upload was performed.

## Implementation

In 10% of fitting supervised events, independently mask the user vector, user bias and user-specific time bias. Mask exposure inputs to reliability gates too, so hidden identity cannot affect scores. Keep candidate, history and shared time terms available. Learn one separate history scale on these pseudo-cold events; apply it to users with zero fitting-impression exposure at deterministic inference. The original history scale remains available to known profiles. Auxiliary historical preference training remains unchanged at V13 weight 2.0.

Mask sampling uses a private seeded Torch generator and does not change initialization or NumPy auxiliary controls. The new scalar starts at the original scale, without consuming RNG. The no-dropout default retains old checkpoint behavior. The main selected checkpoint learns known-profile history scale 0.228859 and cold scale 0.904429. These are internal parameters, not probability multipliers. This changes both profile masking and cold scaling; their separate causal effects were not isolated. Only dropout rate 0.1 was tested.

## Temporal comparisons

| Raw first-user frame | V13 | V14 |
|---|---:|---:|
| January 26 | 0.8150459651 | 0.8161383882 |
| January 25 | 0.7700293538 | 0.7752369666 |
| January 24, broader horizon | 0.7349994137 | 0.7377589749 |

Selected epochs are 2, 6 and 9 respectively. Earlier checkpoints are selected strictly before the next cutoff. Reported complete-frame scores retain all eligible first-user events. All three spectral basis metadata files match V13 exactly. No validation histories/labels enter embedding fitting; zero-exposure status uses fitting impressions only.

The row combiner fits only 35,230 actual January 25 OOT events, excluding 8,734 later first events, with 388 rounds selected on earlier held-out users. The event model uses the same strictly earlier source and 497 rounds. The existing 0.25 factor logit mixture and half event correction remain unchanged. Final mixture AUC is 0.8190372344 before event correction; corrected AUC is 0.8194028105.

Final unseen-fitting-user AUC improves **0.7062846074 → 0.7313715393** on 595 events; seen-user AUC improves **0.8204160263 → 0.8207008460** on 34,543 events. History-length groups 0–5, 6–15 and 16–20 improve by +0.004910, +0.001006 and +0.000226. Subgroup scores are diagnostics, do not average to global AUC and are not independent tests.

## Verification and limits

23 relevant contract tests pass, covering neutral initialization, hidden-profile identity invariance, zero gradients to hidden user parameters, finite cold-scale gradients, cold-only inference routing, legacy history/context behavior, objectives and negative sampling. A read-only checkpoint audit recomputes 118 events including 51 cold events: both V13 and V14 exactly match saved probabilities (maximum difference 0.0), and repeated inference is identical. Runner syntax/help was checked.

These are retrospective model-selection comparisons, not untouched tests. Bootstrap intervals measure sampling uncertainty only, not repeated search or evaluation population/time shift. The cold subgroup is small. Private/public leaderboard AUC and ranking remain unknown. V13 and earlier submissions are retained.

Refit V14 on all 6,868,240 labelled training rows for two epochs; apply frozen earlier-period combiners. New verified output: **`submission_techarena_20261102_v14.csv`**. The adjacent verification JSON checks exact header, 529,390 original evaluation rowids exactly once, finite probabilities and [0,1] range. No evaluation labels or evaluation-derived fitting statistics are used; no upload or paid compute.

Reproduce temporal confirmations and fresh full-data output:

```sh
python3 arena/run_v14_checks.py --broad --full
```

The runner gates full refit on a positive paired bootstrap interval and earlier raw improvements. Its constituent commands were executed for these results; the entire fresh runner was not redundantly rerun. Machine-readable selection, hashes and verification: `arena_runs/v14_selected_solution.json`. Models/configs/windows/epoch timings and provenance are under `v14_coldpath_d32_day{24,25,26}_s42`, `v14_cal_coldpath`, `v14_event_coldpath`, `v14_coldpath_final_d32_day26_s42_full`, and `v14_final_combiner`. Comparisons and inference audit are `v14_raw*_vs_v13.json`, `v14_final_vs_v13.json`, and `v14_inference_audit.json`. Experiment records append to `experiments.jsonl`.
