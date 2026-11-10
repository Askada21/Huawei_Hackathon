# V25 signed interaction graph

V25 reached **0.823300224** local ROC AUC versus **0.821201318** for V24. The paired event bootstrap interval for the difference was [0.001602950, 0.002499842] on the same 35,138 January 26 events. The target is 0.834673 and the local gap is 0.011372776. The official score is unknown.

The fourth member uses a graph with positive clicks and a negative weight of 0.1 for shown candidates with no click. Both sources come from fitting data. The members receive equal weights in logit space after their own calibration. The signed member was selected as a complement after screening five alternatives at a fixed 25% weight. Its raw gains against V24 on January 24, 25, and 26 were +0.015881195, +0.011188588, and +0.002126224. The interval reflects sampling on this repeatedly used validation frame.

Verified CSV `submission_techarena_v25_reproduced_20261103T113431Z.csv` contains 529,390 original evaluation IDs and finite probabilities in [0,1]. Reproduce with `python3 arena/run_v17_checks.py --variant signed --broad --full` using the V24 artifacts. Comparisons and configurations are in `arena_runs/v25_checks`. Verification is in `arena_runs/v25_selected_solution.json`.

## Additional screens

The standalone January 26 screens were 0.817105357 for activity history (V26), 0.816850164 for a wider time-context rank (V27), 0.814591837 for a nonlinear affinity head (V28), and 0.817092107 for sequential auxiliary learning (V29). None matched the contribution from the signed graph at a fixed 25% logit weight against V24.

The lighter signed graph with weight 0.025 (V30) reached 0.822904773 after calibration and a corrected 25% blend with V24. The first V30 runner used V19 as a blend source while comparing against V24. Those comparisons are superseded by the files in `arena_runs/v30_corrected`. Independent fitting artifacts were reused.
