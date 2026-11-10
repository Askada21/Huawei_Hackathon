# V12 graph-degree smoothing: rejected at main raw screen

In the V8 training recipe, adding degree 10 to both graph normalization denominators before spectral initialization scores **0.7769806068 raw AUC**, versus **0.8142925899** for V8 on identical January 26 first-user events. The selected checkpoint is epoch 2. Stop this branch before the earlier fold, calibration or full refit. No new CSV is generated. Reducing localization of the basis was insufficient to improve predictive quality; this is not evidence of a broken baseline graph.

An earlier degree-10 experiment in a different, older model had also been weak (0.752190). The new test keeps V8's objective, gates, auxiliary observed controls, learning rate and contextual architecture fixed, so the current comparison is a controlled initialization ablation. Only degree 10 was tested; no degree sweep.

Added optional `--spectral-degree-reg` to the history trainer. The neutral value explicitly retains the original `reg0` cache key so old V8 initializations remain exact. Float `reg0.0` would otherwise construct a duplicate cache; the initial V13 exploratory run exposed this and was excluded in favor of RNG/basis-controlled runs. Existing caches and models were preserved.

Config, timings, epoch log and checkpoint: `arena_runs/v12_graphreg_d32_day26_s42`. Read-only comparison: `v12_raw26_vs_v8.json`. Reproduce the screen with `python3 arena/run_v12_checks.py`; the runner stops if the main raw comparison declines. Constituent commands were executed; the full fresh runner was not redundantly rerun. History/context/objective/sampling tests and runner syntax/help passed. No upload, paid compute or evaluation-label access.
