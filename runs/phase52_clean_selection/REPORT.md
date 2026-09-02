# Phase 52 Clean Selection Report

## Locked result

- Configuration: **C**
- Seed: **0**
- Checkpoint: `C:\Users\chand\OneDrive\Desktop\DepthWizard\runs\phase51_corrected_unet\checkpoints\C_seed_0_best.pt`
- Threshold: **0.60**
- Selection data: Copenhagen only

## Copenhagen evidence

The selected checkpoint achieved locked-threshold IoU `0.5353`, Dice `0.6914`, precision `0.6403`, recall `0.7712`, and predicted foreground `48.59%`. Full eight-checkpoint results are in `COPENHAGEN_SELECTION.csv`.

## New York protocol

Phase 51 generated New York metrics before final selection. Those older values were not used here. The authoritative clean result from the locked checkpoint and locked Copenhagen threshold is in `NEW_YORK_FINAL.csv`.

- IoU: `0.1365`
- Dice: `0.1974`
- Precision: `0.8873`
- Recall: `0.1470`
- Predicted foreground: `5.49%`
- Matched / missed buildings: `672 / 1760`

## Height and downstream comparison

Complete New York height diagnostics are in `HEIGHT_RESULTS.csv`; mean building MAE is `55.4650` and mean bias is `55.4636`. The fixed existing calibration/height path was run on one NYC tile for Phase 29 Config A versus the locked Phase 52 model. No production code, PeakRecoveryMLP, DSM, nDSM, DTM, or viewer code was modified. The generated visual comparison is evidence for one demo tile, not a complete 3D acceptance claim.

## Verdict

**PHASE52_PARTIAL_SUPPORT**

The corrected model is non-collapsed and Copenhagen selection is complete, but this evaluator does not establish complete multi-tile 3D improvement or human visual acceptance across the city.
