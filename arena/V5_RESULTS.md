# V5: supervised comparisons of displayed alternatives

**Selected local AUC 0.8166685543**, V3**0.8151557921**, on identical 35,138 first-user events/351,380 rows after Jan26. Last known official public V3 AUC remains**0.808665**. No new official score is known and no upload was performed. The target0.834673 is not achieved or guaranteed.

## Mechanism and controls

The existing event softmax already learned candidate competition. V5 changes how incorrect rankings are penalized: each event's winner is compared individually against all ten losing outcomes, including a fixed-score no-click option. No-click events are retained and drive all ten candidate scores downward. Pair loss scaling 20/11 matches the softmax gradient at zero logits for every possible event outcome. Loss values themselves are not directly comparable between objectives.

The selected variant also replaces catalogue controls in the historical-click auxiliary term with **actual fitting-period clicked=0 impressions of the same user**. No unknown impression is labelled negative. Users without an observed negative pool are skipped for that auxiliary term; same-item contrasts are omitted. Historical positive interactions remain deduplicated. The spectral basis, dimensions, initial vector scale, global gates, learning rate, optimizer, regularization, and auxiliary coefficient remain fixed. See `V5_OBJECTIVE_AUDIT.md` and the source configurations for exact details.

## Experiments

| Raw factor experiment, Jan26 | AUC |
|---|---:|
| V3 event softmax | 0.811248 |
| Winner-versus-losers pairwise | 0.812453 |
| Equal softmax/pairwise hybrid | 0.811847 |
| Softmax plus observed cross-event ranking | 0.811836 |
| Separately learned, regularized history contexts under softmax | 0.810198 |
| **Pairwise + same-user observed auxiliary controls** | **0.813080** |

The first pairwise experiment's raw-AUC delta against V3 has paired sampling interval[0.000915,0.001437]. This justified bounded hybrid checks, not a loss-weight sweep. Neither hybrid beat pairwise. The separate history estimator did not improve the baseline and was rejected.

## Other temporal windows

| First-user cutoff | V3 comparable raw factor | Selected raw factor |
|---|---:|---:|
| Jan25,43,964 events across2 days | 0.752715 | 0.761332 |
| Jan24,47,928 events across3 days | 0.710126 | 0.724477 |

The pure pairwise variant reaches0.763215 on Jan25; observed auxiliary controls are not uniformly better than pure pairwise on every window. The selected combination nevertheless improves both earlier windows against the incumbent and has the best complete Jan26 pipeline. Earlier checkpoints are selected only on their own day, strictly before the next cutoff. Broader-window scores include all eligible first events, not hand-picked easy rows. These retrospective comparisons are model-selection/backtest evidence, not untouched future tests.

## Complete selected pipeline

1. Train the factor model using pairwise and same-user observed-impression auxiliary losses. Retain V3's softmax-with-outside prediction rule, holding scoring architecture fixed.
2. Fit the430-round row combiner on 35,230 eligible Jan25 OOT events; exclude 8,734 earlier-cutoff first events occurring later. Select rounds using earlier held-out users only.
3. Keep the pre-existing 0.25 factor/0.75 combiner logit mixture: AUC 0.816181.
4. Refit the event component on the same strictly earlier feature source; its selected500 rounds and unchanged half event-mass correction produce**0.816669**. Earlier held-out row AUC rises0.771838→0.772174 with this correction, but those users also select stopping rounds.
5. Refit the selected factor model for two epochs on all 6,868,240 labelled training rows, as selected on Jan26. Freeze the strictly earlier combiners for evaluation inference. No evaluation-derived fitting or hidden labels are used.

The final paired whole-event bootstrap(150 repetitions) gives delta**+0.00151276**, interval**[0.00116638,0.00185245]** against V3. Sampling intervals do not account for architecture selection or temporal shift. In particular, the Jan26 validation sample is weekend-only while evaluation is predominantly weekdays, and final ranking depends on private leaderboard data.

Subgroup improvements are diagnostic only: short-history AUC 0.739557→0.741406, unseen-fitting-user AUC 0.706017→0.713269. They are not substituted for overall AUC.

## Delivery

Recommended new candidate: `submission_techarena_20261102_v5.csv`. Its verification JSON checks the exact header,529,390 predictions, complete unique evaluation rowids, and finite probabilities in[0,1]. V3 remains preserved as the officially scored reference; V5's official score is unknown.

From the project root, reproduce training, temporal confirmation, full refit and a new verified CSV:

```sh
python3 arena/run_v5.py --broad
```

The command writes a new timestamped directory and never overwrites delivered CSVs. It reuses compact source arrays/caches and uses the previous prediction files only for comparisons. Configurations, epoch logs, calibration/event provenance, paired comparisons and code/model hashes are recorded under `arena_runs/v5_*` and `v5_selected_solution.json`.
