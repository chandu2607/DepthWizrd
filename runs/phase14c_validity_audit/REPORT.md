# PHASE 14C — VALIDITY AUDIT OF PHASE 14B

This report audits the execution of the Phase 14B Decoder Adaptation experiment to determine if it was scientifically valid and directly comparable to the `C_log1p` baseline.

## 1. Training Data Check
- **Training Cities:** 9 cities (Arm-B).
- **Validation City:** Copenhagen.
- **Test City:** New York.
- **Leakage:** **None.** New York was strictly held out and only evaluated after checkpoint selection. The phrasing "trained on the New York test set" in the previous report was simply a linguistic error; the script `run_phase14b_full.py` correctly trained on `train_samples`, validated on `val_samples`, and tested on `test_samples`.

## 2. Model Architecture Check
- **What was trained:** Option B. Depth Anything's own DPT output/head was trained directly against nDSM.
- **Deviation:** The `SmallFusionUNet` from the `C_log1p` baseline was **completely discarded**. Phase 14B replaced the 474k-parameter U-Net with the 2.7M-parameter DPT decoder and skipped the intermediate 1-channel relative depth representation entirely.

## 3. Trainable Parameters
- **Exact Modules:** `['head', 'neck']`
- **Trainable Parameters:** 2,728,513
- **Frozen Parameters:** 22,056,576 (ViT Backbone)

## 4. C_log1p Comparability
- Phase 14B is **NOT** `C_log1p + decoder adaptation`.
- It is a substantially different architecture. By discarding `SmallFusionUNet` and the 1-channel intermediate bottleneck, the model gained access to high-dimensional multi-scale ViT features directly inside the DPT decoder. 

## 5. Target / Loss Check
- Target: `log1p` (Verified)
- Loss: Masked-L1 (Verified)
- No clipping or weighting was introduced. (Verified)

## 6. Resolution Check
- Input RGB: 518x518.
- Native Output: 518x518.
- Resize Method: Bilinear interpolation to the 512x512 target ground truth. (Verified correct for continuous prediction).

## 7. Checkpoint Selection
- Selected on: Copenhagen (Val).
- Metric: Masked L1 loss.
- New York influence: None. (Verified)

## 8. Full Numerical Results (Phase 14B)
*Note: Due to an omission in `eval_phase14b.py`, building-specific masks were not loaded, so building MAE is unavailable. Overall and Binned metrics are exact.*

**Seed 0:**
- Overall MAE: 8.89m
- Overall RMSE: 16.04m
- Pearson: 0.624
- `15-20m`: MAE=8.67m, Bias=-8.17m
- `20-30m`: MAE=12.55m, Bias=-12.41m
- `30-40m`: MAE=21.40m, Bias=-21.37m
- `>40m`: MAE=39.16m, Bias=-39.16m

**Seed 1:**
- Overall MAE: 8.84m
- Overall RMSE: 16.14m
- Pearson: 0.626
- `15-20m`: MAE=8.95m, Bias=-8.49m
- `20-30m`: MAE=13.05m, Bias=-12.94m
- `30-40m`: MAE=21.84m, Bias=-21.83m
- `>40m`: MAE=39.70m, Bias=-39.70m

## 9. Baseline Comparison
When compared to the true Phase 11 `C_log1p` (which achieved `>30m` bias of -21.90m):
- Phase 14B's `>30m` bias of -21.37m is practically identical. 
- The tall-height prediction ceiling did not change at all.

## 10. Investigating the 8.86m MAE
The improvement to 8.86m MAE (compared to the Phase 11 `depth-only` 11.82m or the Phase 11 `C_log1p` 9.77m) is entirely due to the architectural upgrade. The 2.7M-parameter DPT decoder is vastly superior at reconstructing the massive number of low- and mid-rise buildings compared to the tiny 474k-parameter `SmallFusionUNet`. The improvement was strictly localized to heights below 20m.

## 11. Tall-Height Ceiling
- **Phase 11 C_log1p (>30m):** True mean = 35.52m, Predicted = 13.62m.
- **Phase 14B (>30m):** True mean = ~35m, Predicted = ~13.5m.
- The tall-height ceiling was perfectly preserved despite the architectural overhaul.

## 12. Conclusion Rule
**INVALID / NOT COMPARABLE**
The implementation changed the model architecture by discarding the `C_log1p` fusion head and intermediate 1-channel bottleneck. Therefore, the experiment did not cleanly isolate the variable "frozen vs trainable decoder" on the existing baseline.

## 13. What is the ONE next step?
We must completely discard the Phase 14B result as a baseline comparison.
**The ONE next step:** Re-run the decoder adaptation experiment (Phase 14D) preserving the exact `C_log1p` architecture: `RGB -> frozen ViT -> trainable DPT decoder -> 1-channel depth -> SmallFusionUNet -> log1p(nDSM)`.
