# Phase 52 Clean Selection Protocol

Phase 51 generated New York metrics during training. Those earlier NY numbers are retained as post-hoc/contaminated-for-selection artifacts and were not used here.

This script first evaluated all eight best checkpoints on Copenhagen only. It rejected near-all-foreground outputs, selected the best non-collapsed checkpoint by Copenhagen IoU with Dice and precision/recall reported, then selected the threshold using Copenhagen only. The checkpoint and threshold were locked before the complete New York evaluation.

All completed Phase 51 checkpoints used batch size 16, as recorded in their execution provenance. No retraining occurs in Phase 52.
