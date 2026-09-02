# Phase 51 Report

## Decision

**UNET_CORRECTION_PARTIAL_SUPPORT**

The corrected training experiment used the full 937-tile training split for B/C/D, seeds 0 and 1, and Copenhagen-only checkpoint/threshold selection. New York was evaluated only after locking the candidate.

The historical A control remains a 32-tile DEBUG/HISTORICAL control. Corrected targets use nearest-neighbor binary masks.

Best Copenhagen candidate: Config C, seed 1, validation IoU 0.5353. Selected threshold: 0.40. New York IoU: 0.2673; Dice: 0.3885; precision: 0.7802; recall: 0.3226; foreground: 17.96%.

The result is partial support because this script does not claim downstream 3D improvement. 3D impact remains gated for human review.
