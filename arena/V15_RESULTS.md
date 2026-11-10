# V15: 31-leaf row combiner — not selected

The tested candidate reaches January 26 global row AUC **0.8195781391**, versus V14 **0.8194028105** on identical 35,138 first-user events / 351,380 rows. Difference **+0.0001753286**. Paired whole-event bootstrap, 150 replicates, seed 42: **[-0.0000035268, +0.0003148817]**, including zero.

The earlier January 24→25 temporal comparison does not confirm the gain. On identical 43,964 January 25 first-user events, fixed factor/row mixture AUC is **0.7815954121** with 31 leaves versus **0.7816950769** with 7 leaves, difference **-0.0000996648**. These earlier scores exclude event correction, identically for both candidates; they must not be compared directly to January 26 final AUC.

**Decision: retain V14 and its existing verified CSV.** Do not create another submission for this weak result. V14 selected local AUC remains 0.8194028105; official V14 AUC remains unknown. Last reported official V3 public AUC is 0.808665. No upload or paid compute.

## Controlled change

Only the row combiner's maximum leaf count changes from 7 to 31. Keep min leaf size 500, L2 30, learning rate 0.03, 500-round ceiling and earlier-user stopping. Factors and their features remain frozen V14 artifacts. Main combiner training uses only 35,230 actual January 25 OOT events, excluding 8,734 later first events. Earlier confirmation uses only actual January 24 OOT events from the checkpoint selected before January 25, excluding every later event. Both earlier 7/31-leaf models use the same input rows and held-out-user split.

Mixture weights remain 0.25 factor / 0.75 combiner logits, followed by fixed half event correction. The V14 event model is reused because its event features, training and probabilities are independent of row-combiner complexity. Its historical held-out 7-leaf row diagnostic is not claimed as confirmation for V15. Event order/schema validation protects every blend and event correction; the apply tools do not read prediction labels.

Exposed `--leaves` in `calibrate.py`, preserving default 7. Existing output directories now raise instead of overwriting models. Fifteen calibration/event contract tests pass; runner syntax/help was checked. Existing V14 submission bytes remain unchanged.

## Reproduction and evidence

```sh
python3 arena/run_v15_checks.py --submission
```

This reuses frozen V14 factor feature caches, fits new combiners in a fresh directory, and generates a new CSV only if the paired main bootstrap interval and earlier fixed-mixture comparison support an improvement. The reported result fails those gates. No factor refit is needed. Constituent commands were run; the complete fresh runner was not redundantly rerun.

Artifacts: `arena_runs/v15_cal31_day26`, `v15_screen7_day25`, `v15_screen31_day25`, `v15_final_vs_v14.json`, `v15_earlier_vs7.json`, and `v15_decision.json`. All configs, selected rounds, model files, input paths and experiment records are retained. Main and earlier temporal frames remain repeatedly examined model-selection data. Bootstrap only addresses sampling uncertainty, not search bias or evaluation shift.
