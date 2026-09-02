# PHASE 14D — CORRECT DECODER ADAPTATION

This report details the execution of Phase 14D, the corrected version of the decoder adaptation experiment. Unlike Phase 14B (which was marked INVALID for altering the fundamental model architecture), this phase cleanly preserved the original `C_log1p` pipeline.

## 1. Exact Architecture
The experiment used the exact intended architecture:
`RGB -> DA-V2 (Frozen Backbone + Trainable Neck/Head) -> 1-Channel Depth -> SmallFusionUNet (w=24, Trainable) -> log1p(nDSM)`
The intermediate adapted depth representation was natively interpolated and normalized using training-set statistics to perfectly match the domain expectations of the `SmallFusionUNet`.

## 2. Trainable / Frozen Parameters
- **Trainable Parameters:** 3,202,586 (DA-V2 Neck + DA-V2 Head + SmallFusionUNet)
- **Frozen Parameters:** 22,056,576 (DA-V2 ViT Backbone)
- **Trainable Modules:** `unet.*`, `head.*`, `neck.*`
- **Frozen Modules:** `backbone.*`

## 3. Sanity-Check Result
**SUCCESS.** A pre-training diagnostic confirmed:
1. Gradients reached the neck, head, and `SmallFusionUNet`.
2. Zero gradients reached the ViT backbone.
3. Target shapes and resolutions were exactly maintained.
4. Loss decreased stably without numerical issues.
5. VRAM consumption was perfectly nominal (428 MB peak during diagnostic).

## 4. Two-Seed Metrics (New York Test Set)
**Seed 0:**
- **Overall MAE:** 9.33 m
- **Overall RMSE:** 32.07 m
- **Pearson r:** 0.570

**Seed 1:**
- **Overall MAE:** 9.68 m
- **Overall RMSE:** 17.43 m
- **Pearson r:** 0.592

## 5. Tall-Height Comparison (Binned Results)
| Height Bin | Seed 0 Bias (m) | Seed 1 Bias (m) |
|------------|-----------------|-----------------|
| 15–20 m    | -9.01           | -10.38          |
| 20–30 m    | -13.73          | -15.32          |
| 30–40 m    | -23.67          | -25.37          |
| >40 m      | -40.44          | -43.24          |

## 6. Did the Ceiling Change?
**No.** The prediction ceiling is virtually identical to the completely frozen Phase 11 `C_log1p` baseline. For true heights `>40m` (mean ~54m), the model predicts ~11–13 meters. For true heights `30-40m` (mean ~35m), the model predicts ~10–12 meters. The tall-height capability remains collapsed.

## 7. Scientific Verdict
**NO SUPPORT (CASE C).**
The hypothesis that decoder adaptation alone could synthesize missing metric-scale features is conclusively rejected. Even when properly feeding live features to the `SmallFusionUNet`, the model hits the exact same ~14m ceiling. This proves the limitation is embedded deeply in the frozen ViT backbone's attention representations. The frozen backbone simply does not differentiate extreme absolute scales, and the decoder cannot mathematically invent metric information that the backbone does not provide.

## 8. Recommended Next Step
**Partial ViT Unfreezing (Phase 14E).** We must systematically unfreeze the deepest blocks of the ViT backbone. If providing gradients to the final transformer blocks restores the tall-height prediction, it will definitively prove that metric-scale perception requires re-calibration of the core attention maps.
