# V15: complementary nonlinear row combination

Keep all V14 factor models, features, fixed 0.25 factor logit weight and fixed half event correction. Test a single row-combiner complexity change: 31 leaves instead of 7, with the same 500-round ceiling, earlier held-out-user stopping, min leaf size 500, learning rate 0.03 and L2 30. Hypothesis: richer interactions between factor scores, history and fitting-only exposure statistics improve ranking. No leaf-count grid.

Main fit uses actual January 25 OOT events only, excluding January 26 onward. Reuse the V14 event model because its features and fitted event probabilities are independent of the row combiner; its held-out 7-leaf row diagnostics remain historical, not evidence for the new combiner. Compare V15 versus V14 on identical January 26 rows, with paired event bootstrap. Confirm the fixed factor/row mixture on January 25 using separate 7- and 31-leaf combiners trained on actual January 24 OOT events and a January-24 checkpoint selected before January 25. All feature schemas/event orders must agree. No new factor training is needed. Generate a new CSV only for a supported improvement, no upload.

Expose the leaf-count option in `calibrate.py` without changing its default. Also refuse existing output directories to prevent accidental artifact overwrite. Preserve all prior models and submissions.
