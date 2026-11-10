# V24 three-seed ensemble

V24 reached **0.821201318** local ROC AUC versus **0.821020012** for V22 on the same 35,138 January 26 events. The paired event bootstrap interval for the gain is [0.000054799, 0.000313372]. The interval describes sampling variation on this model-selection frame.

The ensemble gives equal logit weights to seeds 42, 43, and 44. Each model uses its own earlier-period calibration and event correction. Raw AUC gains on January 24, 25, and 26 were +0.000712567, +0.000829015, and +0.000149783. The added model was refitted for two epochs. Thirteen existing tests passed.

Verified CSV `submission_techarena_v24_reproduced_20261103T112445Z.csv` contains 529,390 original evaluation IDs and finite probabilities in [0,1]. Official AUC is unknown. Reproduce with `python3 arena/run_v17_checks.py --variant seed44 --broad --full` using the existing V19 and V22 artifacts. Configurations and comparisons are in `arena_runs/v24_checks`. Verification and selection are in `arena_runs/v24_selected_solution.json`.
