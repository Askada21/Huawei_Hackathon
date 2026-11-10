# V37 seven-member ensemble

V37 reached **0.825159631** local ROC AUC, up from **0.824546812** for V31. The gap to 0.834673 is **0.009513369**. Official AUC is unknown.

| Candidate | January 26 local ROC AUC |
|---|---:|
| V31 | 0.824546812 |
| V32 | 0.824901929 |
| V34 | 0.824436452 |
| V35 | 0.823972232 |
| V36 | 0.824876908 |
| V37 | 0.825159631 |

V32 weights observed nonclicks by the geometric mean of smoothed user and item relative click odds, clipped and normalized within fitting data. V36 adds the lighter signed graph. V37 gives the five V31 models and both new models equal logit weights. Against V32, gains before calibration on January 24, 25, and 26 were +0.005094066, +0.002340607, and +0.000132811. The full calibrated comparison gave +0.000257702 with paired event bootstrap interval [0.000151174, 0.000368134]. This interval describes sampling variation on the reused model-selection frame.

V34 fits a joint row calibrator on earlier cached model predictions. V35 fits a grouped LambdaRank model and reallocates part of V31 event mass. Both gained on the earlier raw reference and declined against the full January 26 V31. The 48-factor signed graph V33 declined at the raw screen.

Verified CSV `submission_techarena_v37_join.csv` contains 529,390 original row IDs and finite probabilities in [0,1]. V31's CSV hash still matches its verification record. Reproduce with `OPENBLAS_NUM_THREADS=1 python3 arena/run_v37_join.py --out v37_repro --full` using the existing V31, V32, V36 and V30 artifacts. The V36 model refit and validation can be reproduced with `python3 arena/run_v36_extend.py --out v36_repro --full`. V32 uses `python3 arena/run_v17_checks.py --variant adaptive --broad --full`.
