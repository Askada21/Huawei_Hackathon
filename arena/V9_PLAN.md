# V9: bucket-specific preference gates

One small ablation of V8: add eight regularized gate offsets, two for each six-hour bucket. They separately scale the user vector and pooled history before V8's contextual transform. Offsets start at zero without consuming initialization randomness, preserving both V8's initial scores and contextual basis. Keep rank, objective, negatives, seed, optimizer and mixture weights fixed.

Compare raw factors on January 26 and January 25, with the earlier checkpoint selected strictly before January 26. Fit new row/event combiners on actual January 25 OOT events only, then compare the complete pipeline against V8 on identical January 26 events. Select only if evidence supports an improvement; otherwise preserve V8 and do not create another submission. Whole-event bootstrap quantifies sampling uncertainty, not repeated-selection bias.

The initial `v9_timegate` run consumed random numbers constructing the embedding before zeroing it, changing contextual initialization. Treat it as superseded exploratory output; the RNG-neutral `v9_controlled` experiment is the comparison of interest. Do not select the uncontrolled run.
