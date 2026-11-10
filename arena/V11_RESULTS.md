# V11 contextual auxiliary preference learning: rejected

V8 uses context-dependent preferences for supervised impressions but static user vectors for auxiliary historical clicks. Tested transforming auxiliary user vectors with the click's known six-hour bucket. Auxiliary controls now come exclusively from actual nonclicked fitting impressions of that same user and bucket. Users/buckets with no such controls are skipped; identical positive/control items remain masked. This changes both auxiliary scoring and sampling, so their individual effects are not identified. The baseline was not a proven implementation bug.

| Raw factor, identical first-user frame | V8 | V11 |
|---|---:|---:|
| January 26 | 0.8142925899 | 0.8102798536 |
| January 25 | 0.7624680071 | 0.7566877264 |

January 26 selects epoch 2 and stops after four non-improving epochs. January 25 selects epoch 12, the predefined maximum; it was still improving, so this is a budget-limited result, not a claim of convergence. That checkpoint is selected only using events actually before January 26. Both complete-frame raw comparisons decline. Skip calibration, event correction, full-data refit and new submission; this does not prove that every contextual auxiliary model fails or rule out complementary ensembles under other designs.

The same-user/bucket control pool covers 802,242 / 881,345 distinct fitting clicks before January 25 (91.02%) and 881,618 / 946,227 before January 26 (93.17%). Narrower control coverage is a tradeoff of this hypothesis, not evidence about evaluation labels. All statistics and models use fitting-period inputs only. V8 architecture, initialization, seed, loss weights, regularization and optimizer are unchanged; rejected V9/V10 flags are disabled.

Two new tests cover exact user/bucket membership, clicked/future-row exclusion and auxiliary gradient routing to the selected context. Existing sampling, context, time-history, history and objective checks also pass: 22 tests total. Syntax and reproduction `--help` were checked. Existing configs/predictions retain their inference behavior.

Decision: retain the unchanged `submission_techarena_20261102_v8.csv`, local full-pipeline AUC 0.8174909942. Its official AUC is unknown; last reported official V3 public score remains 0.808665. No upload or paid compute was used. Repeated retrospective validation remains a limitation.

Models/configs/epoch logs/runtime: `arena_runs/v11_contextaux_d32_day{25,26}_s42`. Comparisons: `v11_raw25_vs_v8.json`, `v11_raw26_vs_v8.json`. Coverage: `v11_aux_coverage.json`. Decision/code hashes: `v11_decision.json`. Training and decision records append to `experiments.jsonl`.

Reproduce in fresh artifacts:

```sh
python3 arena/run_v11_checks.py
```

The runner stops before combiners if both raw comparisons decline. Its constituent training/comparison commands were run for this report; the entire fresh runner was not redundantly rerun. V8 comparison files are read-only references.
