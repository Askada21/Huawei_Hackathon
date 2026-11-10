# V9: small temporal-gate ablation

Implemented eight regularized offsets to let the user-profile/history balance vary by six-hour bucket. Zero initialization preserves V8's initial predictions and contextual basis without consuming random numbers. The feature is opt-in (`--gates --temporal-gates`); old configurations retain their behavior. Four context-contract tests and twelve history/objective tests pass.

| Identical validation frame | V8 | V9 controlled |
|---|---:|---:|
| January 25 raw factor | 0.7624680071 | 0.7623488168 |
| January 26 raw factor | 0.8142925899 | 0.8142599391 |
| January 26 complete pipeline | 0.8174909942 | 0.8175231062 |

The final difference is +0.0000321119. A paired whole-event bootstrap (150 replicates, seed 42) gives a 95% percentile interval of **[-0.0000495744, +0.0001027678]**. The interval includes zero, and both raw temporal comparisons decline. This is insufficient evidence to replace V8. The bootstrap covers sampling uncertainty only, not repeated model selection or temporal shift.

The earlier model fits only events before January 25, and checkpoint selection excludes January 26 onward. Both combiners fit only 35,230 actual January 25 OOT events, excluding 8,734 later first-user events. V9 selects 344 row-combiner rounds and 497 event rounds on earlier held-out users. The existing 0.25 raw logit mixture and half event correction are unchanged. No validation labels enter factor training, no evaluation labels are available, and no upload was performed.

Decision: retain `submission_techarena_20261102_v8.csv`; do not spend a submission attempt or full-data refit on this weak result. Its local AUC remains 0.8174909942. Last reported official score remains V3's 0.808665; V8's official AUC is unknown. The superseded `v9_timegate` initialization-confounded exploratory run is preserved but excluded from selection.

Artifacts: `arena_runs/v9_controlled_d32_day{25,26}_s42`, `v9_cal_controlled`, `v9_event_controlled`, `v9_raw25_vs_v8.json`, `v9_final_vs_v8.json`, and `v9_decision.json`. Epoch logs, full configs, training windows, elapsed seconds and paths are stored alongside models and in `arena_runs/experiments.jsonl`.

Reproduce the controlled experiment in a fresh directory (existing V8 predictions are required only for read-only comparisons):

```sh
python3 arena/run_v9_checks.py
```

The runner's syntax and `--help` path were checked; the constituent training, exports, calibration, fixed mixture, event correction and comparison commands were run for the reported result. The complete fresh runner was not redundantly rerun.

Recommended next evidence: score the already verified V8 submission on the official platform, manually or with explicit upload authorization, before more local micro-tuning. The remaining local-versus-official shift is a more material uncertainty than these gate offsets.
