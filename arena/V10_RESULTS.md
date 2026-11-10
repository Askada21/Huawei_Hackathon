# V10 contextual-history experiment

Implemented 16 learned current-bucket × historical-click-bucket weights. The masked history vector uses L2-normalized weights, starting at exactly the V8 pooling rule. No RNG is consumed by initialization; old configurations remain unchanged. The option is `--time-history` with baseline pooling. Padding, empty histories, exact neutral initialization, time sensitivity and gradients are covered by tests. All 13 context/history/time-history tests passed; reproduction runner syntax/help was checked.

| Same first-user validation frame | V8 | V10 |
|---|---:|---:|
| January 25 raw factor | 0.7624680071 | 0.7624682435 |
| January 26 raw factor | 0.8142925899 | 0.8142928465 |
| January 26 final pipeline | 0.8174909942 | 0.8175194731 |

Final delta: **+0.0000284789**. Paired whole-event bootstrap, 150 replicates, seed 42: **[-0.0000278910, +0.0000672684]**. This includes zero. The learned history-bucket logit offsets have maximum absolute magnitude 0.003073 at the main selected checkpoint, so the tested regularized model keeps the weights nearly uniform. This does not rule out other time-conditioned representations, but it gives no useful evidence for replacing V8.

Decision: retain V8, no full refit or new submission. All old submissions remain intact. V8 local AUC remains 0.8174909942, and its official score is unknown. Last reported official V3 public AUC is 0.808665. No platform upload was performed.

Training follows V8 settings: 32-dimensional rank-four contextual factors, observed-impression pairwise objective, same-user observed auxiliary negatives, regularization 1, seed 42. Earlier checkpoint selection is strictly before January 26. Row/event combiners train only on 35,230 actual January 25 OOT events, excluding 8,734 later first events. The existing 0.25 factor logit mixture and half event correction remain fixed. Comparisons use all 35,138 January 26 first-user events / 351,380 rows, with exact event alignment. Retrospective validation and bootstrap cannot account for model-search bias or evaluation time/population shift.

Configs, windows, epoch logs, elapsed seconds, feature provenance and models are in `arena_runs/v10_historybucket_d32_day{25,26}_s42`, `v10_cal_historybucket`, and `v10_event_historybucket`. Raw/final comparisons are `v10_raw25_vs_v8.json`, `v10_raw26_vs_v8.json`, and `v10_final_vs_v8.json`; decision/code hashes are in `v10_decision.json`. Standard experiment records append to `experiments.jsonl`.

Reproduce in a fresh directory:

```sh
python3 arena/run_v10_checks.py
```

The constituent commands were run for these results; the complete fresh runner was not redundantly rerun. Existing V8 predictions are read-only comparison references. No evaluation labels or evaluation-derived fitting statistics are used.
