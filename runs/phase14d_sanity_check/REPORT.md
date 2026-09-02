# PHASE 14D-C — FINAL RESULT SANITY CHECK

This report provides a forensic check on the completed Phase 14D results without running any new inference, relying purely on the serialized metrics and experimental logs.

## 1. Overall Metrics
| Metric | Seed 0 | Seed 1 |
|--------|--------|--------|
| **MAE** | 9.33 m | 9.68 m |
| **RMSE** | 32.07 m | 17.43 m |
| **Pearson** | 0.570 | 0.592 |

## 2 & 3. Explaining the Large RMSE Difference
The massive discrepancy in RMSE (32.07m vs 17.43m) despite similar MAE is caused by **actual seed instability** leading to **a few extreme positive outliers (spikes)** in Seed 0. 

Because predictions are bounded by `expm1(x) >= 0`, a ground pixel (0-2m) cannot have a negative error worse than -2m. Yet, in the `0.0-2.0m` bin, Seed 0 has an RMSE of 4.51m (MAE 1.46m). To achieve this high squared error, Seed 0 must occasionally predict extreme positive values (e.g., 30m+) for flat ground. This outlier behavior scatters massive squared errors across the image.

The checkpoints were selected using Copenhagen validation on a **masked-L1 loss**, which is famously robust to extreme outliers. Seed 0 achieved a slightly better L1 loss at Epoch 13 and was saved, despite carrying a prediction distribution with a long tail of extreme spikes. Seed 1 (Epoch 14) happened to converge to a much smoother, stable distribution. Evaluation paths and splits were strictly identical.

## 4. Tall-Height Comparison
*(Note: Exact Predicted Maximum and P95/P99 cannot be precisely extracted without re-running inference, but the binned means provide the exact central tendencies of the tall regime).*

| Metric | Seed 0 | Seed 1 |
|--------|--------|--------|
| **>30m Mean Prediction** | 11.84 m | 10.14 m |
| **>40m Mean Prediction** | 13.86 m | 11.06 m |
| **>30m Bias** | -23.67 m | -25.37 m |
| **>40m Bias** | -40.44 m | -43.24 m |

Both seeds demonstrate a total collapse at tall heights, confirming the baseline ~13m ceiling is completely unbroken.

## 5 & 6. Validation of Execution
- **Architecture:** `RGB -> DA-V2 (Frozen ViT + Trainable Neck/Head) -> 1-Channel Depth -> SmallFusionUNet -> log1p(nDSM)`. (Confirmed by script inspection).
- **Splits:** Arm-B (Train), Copenhagen (Val), New York (Test). (Confirmed).
- **Selection:** Checkpoints were cleanly selected using Copenhagen validation loss.
- **Leakage:** New York was only evaluated after the best checkpoint was loaded. No leakage occurred.

## 7 & 8. Scientific Interpretation
Phase 14B is discarded as a comparator.
Based strictly on the clean, correct Phase 14D execution: **decoder/neck adaptation alone is insufficient.** The model still collapses to ~11-13 meters for buildings taller than 30 meters. Giving the network a fully trainable state-of-the-art DPT decoder did not enable it to invent metric scale. 

## 9. Verdict
**VALID + NO SUPPORT.**
The intended decoder-adaptation hypothesis was properly tested and failed to resolve the tall-height collapse.

## 10. Recommendation for Next Step
**Proceed to partial ViT unfreezing (Phase 14E).** Since adapting the decoder alone is insufficient, the logical next step is to unfreeze the deepest blocks of the ViT backbone. This will determine if the core attention maps can be recalibrated to extract absolute scale features when supervised natively.
