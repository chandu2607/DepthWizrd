# Phase 74 forensic audit for the Phase 73 India terrain pilot

## Status
The benchmark and common-grid pipeline were already proven real and valid before the one-epoch pilot. The problem in Phase 73 was not a missing raster or geospatial mismatch; it was a numerical failure in the learned terrain prediction path.

## Real evidence from the actual run
- Training-region target statistics: mean = 4662.03515625 m, std = 355.37628173828125 m.
- Validation-region target statistics: mean = 260.3289794921875 m, std = 2.054396867752075 m.
- After one training step, the model predicts a near-constant value around 4623.63525390625 m on the Himachal validation crop.
- The actual target mean in that crop is 260.3289794921875 m.
- Validation metrics from the real run: MAE = 4363.30615234375, RMSE = 4363.30712890625, Pearson = -0.11460259146511471.

## Exact root cause
The target is normalized with train-only statistics:

- mean = 4662.03515625
- std = 355.37628173828125

The model learned to output values near the training-region mean space, not the low-altitude validation geometry. The actual one-epoch validation prediction is approximately 4624 m, which is close to the training mean and far from the Himachal target distribution near 260 m. This is a strong domain-shift failure: the validation region has a much lower elevation regime than the training region, and the single-epoch model immediately collapses to the training prior.

## Why the metric code is not the bug
The inverse-normalization path is mathematically correct:

- inverse_normalize(x, mean, std) = x * std + mean

I checked this directly on the training crop. Reconstruction error was exactly zero for the valid pixels, meaning the code that converts normalized outputs back to meter space is not injecting arbitrary scale.

The metric code also passes identity checks and the +1 m / +10 m sanity cases. In other words, the MAE/RMSE formulas are behaving as designed. The catastrophic error is caused by the prediction values themselves, not by a metric mismatch.

## Final verdict
The Phase 73 one-epoch pilot fails because the model collapses to the training-region prior instead of learning the target terrain distribution for a different Himalayan regime. This is a real scientific limitation of the current setup, not an artifact of the evaluation code.

BUG_LOCALIZED
