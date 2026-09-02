# Phase 65 — Training Readiness

## Decision

TRAINING_READY = NO

## Why not

1. Only one real Indian region has been downloaded and aligned at acceptable scientific quality: Uttarakhand.
2. A second and third geographically separated region have been identified but not yet acquired and validated.
3. No high-resolution building reference or building-height labels have been verified for the Indian benchmark.
4. The project has not yet established a valid train/validation/test split with leakage-safe geographic separation.

## Fit-for-future options

The next stage should not be training. It should be: (A) acquire and verify a second and third valid Indian region; (B) secure a proper high-resolution building/height reference; (C) then decide between global + India mixed training, terrain-balanced training, or a dual-branch terrain/buildings pipeline based on baseline failure analysis.
