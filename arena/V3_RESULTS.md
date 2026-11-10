# V3: measured improvement, no leaderboard claim

Selected local row-level ROC AUC: **0.8151557921**, previously **0.8089944032**, on the same 35,138 first-user events / 351,380 rows after January 26. Official evaluation labels remain unknown; the target 0.833803 is not established by these results.

## What changed

The selected factor model learns two global rescaling weights for the user vector and pooled click history. This outperformed the more complex count-dependent gates. We retain 32 factor dimensions, the original spectral initialization, and the past-click auxiliary objective. Two epochs were selected on January 26. For final prediction the factor model is refit on all 6,868,240 labelled candidate rows.

The new LightGBM combiner fits **35,230 first-user OOT events occurring on January 25 only**, excluding all 8,734 later first events from that earlier-cutoff sample. The underlying day-25 factor checkpoint is selected strictly before January 26. The 500-round combiner is frozen for both January 26 validation and evaluation inference. We combine its logits with factor logits using fixed weights 0.75/0.25, then apply the small fixed-half event-click correction. The event model also uses the NEW strictly selected day-25 feature cache and eligible earlier-period labels, with 499 rounds selected on earlier users. Evaluation inference reads no labels.

## Controlled comparisons

| Experiment | Raw factor AUC, January 26 |
|---|---:|
| Previous auxiliary factor model | 0.797924 |
| Frozen item vectors | 0.793613 |
| Anchor regularization | 0.796503 |
| Staged freezing then fine-tuning | 0.796226 |
| Recency-weighted history, 24-hour half-life | 0.796476 |
| Separate frozen history vectors | 0.809007 |
| Candidate-weighted separate history | 0.808831 |
| Count-dependent reliability gates | 0.810601 |
| Separate history + reliability gates | 0.809201 |
| **Two global rescaling weights** | **0.811248** |
| Participation-selected components, 8 dimensions | 0.768814 |
| Participation-selected components, 32 dimensions | 0.805932 |

Spectral localization was observed, but removing localized components did not help. It is not treated as a proven performance bug. Increasing complexity, emphasizing recency, and stronger preservation of initialization did not beat the selected recipe.

| Selected pipeline stage | January 26 AUC |
|---|---:|
| Global-gated factor model | 0.811248 |
| New earlier-period combiner alone | 0.812083 |
| 0.25 factor / 0.75 combiner logit blend | 0.814617 |
| **Plus strict-source fixed-half event correction** | **0.815156** |

Extending the combiner to 1,500 possible rounds and combining OOT January 24+25 did not improve the selected pipeline. We retain the simpler one-day, 500-round combiner. A repeated multi-window run was diagnostic after an initially silent process; it was not counted as independent evidence.

## Other periods and uncertainty

On the January 25 first-user sample, global gating reaches **0.752715**, versus the saved prior factor model's **0.736941**. New day-25 checkpoint selection uses only events before January 26; later labels do not select it. These remain temporal model-comparison results.

The additional fixed-epoch January 24 backtest was mixed: the new two-epoch recipe scored **0.637734**, while the previous three-epoch recipe scored **0.658210**. At an equal two-epoch budget the reference log showed **0.636128**, but this does not establish that the final recipe improves every time window. The shorter fit period and weak spectral structure are important limitations; we did not retune against this result. Its OOT cache was subsequently tested for multi-window calibration, so it is no longer an untouched reserve.

The final 150-replicate paired whole-event bootstrap gives a delta interval **[0.005500, 0.006845]** against the saved baseline. This describes sampling uncertainty only. January 26 was used for architecture/mixture comparisons, and earlier held-out users also selected stopping rounds. Neither is an untouched final test. Historical checkpoint-selection contamination in the previous solution is documented in `arena_runs/v3_audit.md`; the new day-25 source fixes the time boundary for checkpoint selection in BOTH row and event combiners. A provisional 0.815271 result used the older event feature source; it is retained for audit, not used for the delivered CSV.

Subgroups on January 26:

| Group | Previous AUC | V3 AUC |
|---|---:|---:|
| History 0–5 | 0.727663 | 0.739557 |
| History 6–15 | 0.789671 | 0.798196 |
| History 16–20 | 0.827212 | 0.831725 |
| Users absent from fitting events | 0.689759 | 0.706017 |

Subgroup AUCs do not average to global AUC and do not imply causation.

## Reproduce

From the project root:

```sh
python3 arena/run_v3.py
```

This creates fresh timestamped artifacts and a new submission. It retrains the selected V3 factor models and both combiners from the compact source arrays, reproducing strict day-25 feature selection. The baseline predictions are used for comparison only, never as event-model training inputs. Previous models and submissions are preserved. Shared cache writes were made atomic so concurrent experiments cannot expose a partial cache. See `arena_runs/provenance.json` for environment and source hashes.

The recommended artifact is `submission_techarena_20261101_v3.csv`; its accompanying verification JSON checks every original evaluation rowid, exact header, 529,390 predictions, and finite probabilities in [0,1]. No submission was uploaded and no competition attempts were spent.
