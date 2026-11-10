# V31 recency-weighted interaction graph

V31 reached **0.824546812** local ROC AUC versus **0.823300224** for V25 on the same 35,138 January 26 events. The gain was **+0.001246589** with paired whole-event bootstrap interval [0.001040070, 0.001486528]. The interval covers sampling variation on a repeatedly used model-selection frame. The local gap to 0.834673 is 0.010126188. Official AUC is unknown.

The fifth model uses a signed graph with negative weight 0.025 and a seven-day half-life for both clicks and shown candidates with no click. Graph inputs are from each fitting window. The new member receives 20% of the final logit blend with V25. Its raw gains on January 24, 25, and 26 were +0.012077671, +0.006158884, and +0.001237871. The member was promoted after the raw screen. Its final calibration uses earlier-period data and its full refit used two epochs.

The first V31 runner compared the individual model to V25 because the variant was omitted from the ensemble branch. The corrected run in `arena_runs/v31_verified` applies the declared 20% blend and supersedes the first comparison. The V30 blend source was also corrected. See [V25_RESULTS.md](V25_RESULTS.md) for that audit.

All 13 runner contract tests passed. Verified CSV `submission_techarena_v31_reproduced_20261103T113925Z.csv` contains 529,390 original row IDs and finite probabilities in [0,1]. Reproduce with `python3 arena/run_v17_checks.py --variant decay --broad --full` using the V25 artifacts. Run configurations and comparisons are in `arena_runs/v31_verified`. Verification is recorded in `arena_runs/v31_selected_solution.json`.
