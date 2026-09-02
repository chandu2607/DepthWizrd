# Phase 43 Design Document

## Objective
Scientifically validate whether the augmented U-Net (`BuildingConditionedEstimator`) improves building footprint segmentation and downstream 3D city reconstruction without modifying the locked downstream components (`PeakRecoveryMLP`, DSM/nDSM/DTM data).

## Experimental Design
1. **Candidate Configurations**:
   - `Config A`: Baseline without augmentation
   - `Config B`: Geometric transformations (Horizontal/Vertical flips, 90/180/270 rotations, affine jitter)
   - `Config D`: Geometric + Photometric (brightness, contrast, HSV jitter, blur) + Multi-Scale building-aware crops
2. **Protocol**:
   - Model selection strictly on Copenhagen validation IoU.
   - Zero-shot evaluation on New York test split executed exactly once after locking checkpoint.
   - Comparative 3D reconstruction keeping all downstream depth/elevation pipelines identical.
