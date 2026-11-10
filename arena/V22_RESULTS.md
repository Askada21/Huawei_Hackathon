# V22 seed ensemble

V22 reached **0.821020012** local ROC AUC on the 35,138 January 26 first-user events. V19 scored **0.820460142** on the same events. The paired whole-event bootstrap with 150 replicates gives a 95% interval of **[0.000351022, 0.000854452]** for the gain of **0.000559870**. These are local model-selection results. The interval covers sampling variation on this frame.

V22 blends seed 42 and seed 43 with equal weights in logit space. Each member has its own calibration and event correction trained before the validation cutoff. Before calibration, the same fixed blend improved AUC by 0.001481700, 0.002081110, and 0.000594837 on January 24, 25, and 26 respectively. Earlier checkpoint selection stayed within its cutoff. The second member was refitted on the full training set for two epochs. Thirteen existing tests passed.

Verified CSV `submission_techarena_v22_reproduced_20261103T111226Z.csv` contains 529,390 original row IDs and finite probabilities in [0,1]. Official AUC is unknown. The runner reuses V19 artifacts and produces fresh output with `python3 arena/run_v17_checks.py --variant seed43 --broad --full`. Run configurations, comparisons, and timings are in `arena_runs/v22_checks`. Selection and CSV verification are in `arena_runs/v22_selected_solution.json`.
