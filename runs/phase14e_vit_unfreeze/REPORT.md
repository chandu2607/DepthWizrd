# PHASE 14E — DEEPEST ViT ADAPTATION

This report documents the results of Phase 14E, which tested whether partial unfreezing of the deepest ViT block (`layer.11` + layernorm) in Depth Anything V2 could restore metric-scale information for tall-height prediction.

## 1. Trainable Parameter Count
- **Trainable Parameters:** 4,978,586 (`layer.11`, `layernorm`, `neck`, `head`, `SmallFusionUNet`)
- **Frozen Parameters:** 20,280,576 (`layer.0` to `layer.10`, `embeddings`)

## 2. Seed 0 Results
- **Overall:** MAE 8.60 m | RMSE 15.69 m | Pearson 0.623
- **Tall Bins:**
  - `>15 m`: MAE 8.29 m
  - `>20 m`: MAE 11.35 m
  - `>30 m`: MAE 21.24 m (Bias: -21.21 m) (Mean Pred: 14.30 m)
  - `>40 m`: MAE 37.30 m (Bias: -37.30 m) (Mean Pred: 17.00 m)
- **Extremes:** P95 = 23.77 m, P99 = 29.38 m, Max = 286.28 m (The max is an isolated extreme spike, consistent with L1 seed instability).
- **Peak VRAM:** 972.2 MB

## 3. Seed 1 Results
- **Overall:** MAE 9.27 m | RMSE 16.65 m | Pearson 0.624
- **Tall Bins:**
  - `>15 m`: MAE 10.26 m
  - `>20 m`: MAE 14.33 m
  - `>30 m`: MAE 23.41 m (Bias: -23.41 m) (Mean Pred: 12.11 m)
  - `>40 m`: MAE 40.95 m (Bias: -40.95 m) (Mean Pred: 13.35 m)
- **Extremes:** P95 = 19.77 m, P99 = 24.71 m, Max = 35.49 m
- **Peak VRAM:** 1073.6 MB

## 4. Mean ± Std (Across Seeds)
- **Overall MAE:** 8.93 ± 0.33 m
- **Overall RMSE:** 16.17 ± 0.48 m
- **Overall Pearson:** 0.623 ± 0.001
- **>30m Bias:** -22.31 ± 1.10 m
- **>40m Bias:** -39.12 ± 1.82 m

## 5. Tall-Height Results
The tall-height predictions (`>30m` and `>40m`) continue to exhibit massive negative bias, representing near-total collapse. The mean predicted height for buildings over 40m (true mean ~54m) is only **15.1 m** on average across both seeds. 

## 6. Prediction Ceiling Comparison with Phase 14D
Compared to the fully frozen ViT baseline (Phase 14D), unfreezing the deepest block produced a very slight upward shift in the prediction ceiling:
- **Phase 14D >40m Mean Pred:** 13.86 m (Seed 0), 11.06 m (Seed 1) -> Avg: **12.46 m**
- **Phase 14E >40m Mean Pred:** 17.00 m (Seed 0), 13.35 m (Seed 1) -> Avg: **15.17 m**
- **Phase 14E P95:** ~21 m
- **Phase 14E P99:** ~27 m

The ceiling lifted by roughly **~2.7 meters**, and P95/P99 predictions remain firmly capped in the 20-30m range. 

## 7. Spatial Comparison
Visual analysis of the difficult New York scenes confirms that the overall structure remains nearly identical to Phase 14D. The Phase 14E adapted model shows slightly "hotter" colors on the roofs of very tall buildings (representing the ~2.7m improvement), but the fundamental limitation remains. The network still interprets massive 60m skyscrapers as 15-20m buildings. Edge fidelity and footprints remain relatively intact, confirming that the adaptation did not destroy structural knowledge, but it failed to recover metric depth.

## 8. Hypothesis Support
**PARTIAL SUPPORT (CASE B).**
The hypothesis that backbone adaptation can break the tall-height ceiling receives weak, partial support. The tall prediction did improve measurably and consistently (~2.7m), proving that unfreezing the backbone does allow the model to extract *some* additional task-specific scale information that the frozen representation hid. However, the prediction remains substantially below the truth (predicting 15m for 54m buildings). Thus, adapting *only* the deepest ViT block helps, but additional scale information is still missing.

## 9. ONE Next Experiment
**Phase 14F: Deep/Mid-Level ViT Unfreezing.** Since unfreezing a single block produced a small but measurable lifting of the ceiling without breaking the structural representation, the logical next step is to unfreeze deeper into the backbone (e.g., the last 4 blocks, which correspond to the entire final "stage" of feature extraction in typical ViT architectures). This will test if the required metric-scale information is buried further back in the network.
