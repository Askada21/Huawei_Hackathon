# V13: stronger historical preference supervision

**Selected local full-pipeline ROC AUC: 0.8186888601**, versus V8 **0.8174909942**, on identical 35,138 first-user events / 351,380 candidate rows after January 26. Delta **+0.0011978659**. Paired whole-event bootstrap, 150 replicates, seed 42: **[+0.0008731894, +0.0015237238]**. Official V13 and V8 scores are unknown; last user-reported V3 public score remains 0.808665. No upload was performed.

## Change and evidence

Increase the auxiliary historical preference loss weight from 0.5 to 2.0. Its positives are deduplicated fitting/warm-up clicks; controls remain actual same-user nonclicked fitting impressions, with same-item pairs omitted. V8's within-event supervised pairwise objective, rank-32 factors, rank-four time-bucket transform, seed 42, learning rate 0.0024, regularization 1, leading spectral basis and unit normalization remain fixed. V9/V10/V11/V12 options are disabled. One stronger weight was tested, not a weight grid.

| Raw first-user temporal comparison | V8 | V13 |
|---|---:|---:|
| January 26 | 0.8142925899 | 0.8150459651 |
| January 25 | 0.7624680071 | 0.7700293538 |
| January 24, broader horizon | 0.7289144943 | 0.7349994137 |

Main checkpoint: epoch 2. January 25: epoch 6, selected strictly before January 26. January 24: epoch 11, selected strictly before January 25. Full-frame scores include all eligible first-user events; no favorable-row selection. All three basis metadata files match V8 exactly, as does the full-data basis. The first `v13_aux2` exploratory run accidentally used a duplicate float-zero spectral cache; it is preserved but excluded. The reported `v13_aux2_controlled` runs reuse the exact original integer-zero cache. The trainer now explicitly preserves that cache key.

## Complete pipeline and limitations

The row combiner fits only the 35,230 actual January 25 OOT events, excluding 8,734 later first events; 422 rounds are selected on earlier held-out users. The event combiner uses the same strictly earlier source and 372 rounds. Existing mixture weights are unchanged: 0.25 factor / 0.75 row-combiner logits, then half event-mass correction. Fixed mixture AUC is 0.8183372462; event correction gives 0.8186888601.

The final subgroup AUC gains are +0.000375 for history lengths 0–5, +0.000293 for 6–15, and +0.001681 for 16–20. **Unseen-fitting-user AUC declines by 0.006041** on 595 events. January 24 raw unseen-user AUC also declines. Gains are not uniform; keep V8 as a fallback. Evaluation has a different weekday mix, longer horizon and more users absent from fitting events. These retrospective model-selection comparisons and bootstrap sampling intervals cannot guarantee official/private gains or account for model-search bias. No ranking is promised.

The other experiment in this round, degree-10 graph smoothing, declined to raw 0.776981 and was rejected before earlier-fold/combiner/full training; see `V12_RESULTS.md`.

## Delivery

Refitted V13 on all 6,868,240 labelled rows for two epochs. Frozen earlier-period row/event combiners produce `submission_techarena_20261102_v13.csv`. Its verification JSON confirms exact header, 529,390 original evaluation rowids exactly once, and finite probabilities in [0,1]. SHA256: `721269ce7ad358868cc5ab4ae18866260991f69ab4eb7de0e9bf42209c73221f`. V8 and all previous submissions remain unchanged. No evaluation labels or evaluation-derived fitting statistics enter training.

Reproduce temporal confirmations and a new verified full-data submission in fresh artifacts:

```sh
python3 arena/run_v13_checks.py --broad --full
```

Full refit is gated on a positive paired bootstrap interval and improvements on earlier raw frames. The runner's syntax/help and its constituent commands were exercised; the entire fresh runner was not redundantly rerun. Seventeen relevant history/context/objective/sampling tests passed.

Machine-readable selection: `arena_runs/v13_selected_solution.json`. Configs, windows, timings, epoch logs, feature provenance, model hashes and paths are stored under `v13_*` and `experiments.jsonl`. The selected factor checkpoint is `v13_aux2_final_d32_day26_s42_full/model.pt`; row model `v13_cal_aux2/model.txt`; event model `v13_event_aux2/event_model.txt`; final array `v13_final_combiner/evaluation_final.npy`. Recommended next evidence is the official platform score for this candidate; uploading still requires the user's explicit authorization.
