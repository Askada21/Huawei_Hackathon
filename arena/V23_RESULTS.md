# V23 slower training experiment

V23 replaces V22's seed 43 member with a run at learning rate 0.0012 instead of 0.0024. The epoch budget doubled with the existing early stopping. The selected January 26 checkpoint moved from epoch 2 to epoch 4.

Calibrated January 26 ROC AUC reached **0.821280245** versus **0.821020012** for V22. The paired event bootstrap interval for the difference was [0.000098288, 0.000377449]. Earlier raw validation gives the following differences versus V22.

| Date | Replace seed 43 | Add a third equal-weight member |
|---|---:|---:|
| Jan 24 | -0.000520612 | -0.000279596 |
| Jan 25 | -0.000353480 | -0.000223421 |
| Jan 26 | +0.000146796 | +0.000075458 |

V22 remains selected under the earlier-period promotion checks. The three-member blend was screened before calibration. Thirteen existing tests passed. Reproduce V23 with `python3 arena/run_v17_checks.py --variant slow43 --broad --full`. Configurations, timings and comparisons are in `arena_runs/v23_checks`.
