# V13: historical preference objective balance

The selected V8 recipe has used auxiliary weight 0.5 throughout the recent experiments. Its main validation peaks after two epochs and then falls. Test one hypothesis: put greater emphasis on distinct fitting/warm-up click preferences relative to current impression competition, using auxiliary weight 2.0 with the same actual same-user observed controls. This is one objective-weight ablation, not a parameter sweep. Keep regularization 1, seed 42, learning rate 0.0024, initialization, dimensions and contextual architecture unchanged. No V9–V12 features.

Compare January 26, and January 25 with checkpoint selection strictly before January 26. Compare the complete pipeline against V8 if promising; preserve the fixed logit mixture and half event correction. Sampling uncertainty and retrospective selection limitations must be reported. Generate a new full-data submission only for a supported improvement. No automatic upload.
