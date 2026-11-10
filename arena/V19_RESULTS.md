# V17–V19: controlled follow-up to V16

V19 is selected: **0.820460142 local global row ROC AUC**, versus V16 **0.820280590**. Delta **+0.000179553** on the same 35,138 January 26 first-user events. Paired whole-event bootstrap, 150 replicates: **[0.000033187, 0.000303407]**. This interval measures sampling uncertainty on the repeated model-selection frame, not temporal shift or uncertainty from model search.

| Variant | Single change from V16 | Jan 26 raw AUC | Decision |
|---|---|---:|---|
| V16 | Reference | 0.817119751 | Retained as fallback |
| V17 | 48 factors instead of 32 | 0.815357035 | Screen declined |
| V18 | Auxiliary loss weight 4 instead of 2 | 0.817049554 | Screen declined |
| V19 | Confidence denominator +40 instead of +20 impressions | 0.817266146 | Passed raw screens and full comparison |

V19 uses the existing fitting-exposure confidence rule, with four rather than two complete fitting events as the smoothing scale. Dropout, objectives, rank, and blend weights remain unchanged. Raw AUC improved by +0.001126153 on January 25 and +0.000862641 on January 24. Earlier checkpoints were selected strictly before the next cutoff. The row/event combiners use only actual January 25 OOT events, excluding later first events. No new code in the factor trainer; the runner supports all three settings. All 13 history/context/cold-path/shrinkage tests passed in the V19 run.

The runner refitted V19 on all 6,868,240 labelled rows for two epochs and applied frozen earlier combiners. Verified artifact: `submission_techarena_v19_reproduced_20261103T095052Z.csv`, exactly 529,390 unique original evaluation rowids, exact header, finite probabilities in [0,1]. No upload; official score unknown. V16 CSV is byte-for-byte unchanged.

Reproduce in a fresh directory: `python3 arena/run_v17_checks.py --variant shrink40 --broad --full`.

Artifacts: `arena_runs/v19_checks`, `arena_runs/v17_raw26_vs_v16.json`, `arena_runs/v18_raw26_vs_v16.json`, and `arena_runs/v19_selected_solution.json`. Training configs, windows, timings and checkpoints are retained with the runs.

## Follow-up profile smoothing screens

The first-user January 26 screen compared smoothing values 60 and 80 against V19's 40 on the same 351,380 candidate rows. Raw ROC AUC was 0.817250 for 60 and 0.817171 for 80, versus 0.817266 for 40. Both stopped at the raw screen under the runner's promotion rule. The selected full-pipeline score remains 0.820460. Each runner passed 13 existing contract tests.

Comparison artifacts are in `arena_runs/v20_checks/v20_raw26_vs_v19.json` and `arena_runs/v21_checks/v21_raw26_vs_v19.json`. Reproduce with `python3 arena/run_v17_checks.py --variant shrink80 --broad --full` and `python3 arena/run_v17_checks.py --variant shrink60 --broad --full`.
