# PHASE 13C - ORDINAL EVALUATION FORENSICS

## CHECK 1 — GROUND-TRUTH BINNING
**Status:** PASSED (Correct)
- The bins are defined as `np.array([0, 2, 5, 10, 15, 20, 30, 40, np.inf])`.
- `np.digitize` with `right=False` (the default) correctly maps values in `[bins[i-1], bins[i])`.
- A pixel exactly at `40.0` maps to index 8, which `gt_c = np.clip(gt_c - 1, 0, 7)` perfectly clamps to class `7`.
- >30m correctly maps to classes 6 and 7. There are no boundary offset errors.

## CHECK 2 — NEW YORK GROUND-TRUTH COUNTS
**Status:** PASSED (Verified)
- Total valid pixels: 28,311,552
- Class 0 (0-2m): 16,004,264
- Class 1 (2-5m): 331,286
- Class 2 (5-10m): 668,504
- Class 3 (10-15m): 1,526,789
- Class 4 (15-20m): 2,243,798
- Class 5 (20-30m): 3,055,252
- Class 6 (30-40m): 1,590,220
- Class 7 (>40m): 2,891,439
- **Conclusion:** The evaluation set contains nearly 2.9 million valid `>40m` pixels. The test distribution is correct.

## CHECK 3 — PREDICTED CLASS COUNTS
**Status:** PASSED (Verified)
- Class 0: 26,771,712
- Class 1: 0
- Class 2: 577,768
- Class 3: 6,112
- Class 4: 60
- Class 5: 955,900
- Class 6: 0
- Class 7: 0
- **Conclusion:** Predicted class 7 (>40m) and class 6 (30-40m) are both identically zero. The 0% recall is mathematically real.

## CHECK 4 — CONFUSION MATRIX
**Status:** PASSED (Verified)
- Shape is 8x8.
- Row sums perfectly match the GT counts above.
- Column sums perfectly match the Predicted counts above.
- `cm[7, 7]` (true >40m predicted as >40m) is exactly 0. 

## CHECK 5 — RESIZING / SHAPE
**Status:** PASSED (Correct)
- The network outputs 256x256 logits.
- The resizing logic correctly switches to `cv2.INTER_NEAREST` when `target_transform == "classification"`. 
- No class interpolation corruption occurs.

## CHECK 6 — MASKING
**Status:** PASSED (Correct)
- `valid = (np.isfinite(gt)) & (gt != -999.0)` correctly isolates valid ground truth.
- `gt_v = gt[valid]` and `pred_v = pred[valid]` completely strips masked pixels out before metric computation. Invalid pixels are NOT silently assigned to class 0.

## CHECK 7 — ARGMAX
**Status:** PASSED (Correct)
- Standard `torch.argmax(pred, dim=0)` along the channel dimension.
- Nothing algorithmically prevents class 7 from being selected. The logits for class 7 simply never exceed the logits for the dominant classes at any pixel.

## CHECK 8 — DECODED HEIGHT
**Status:** PASSED (Correct)
- The primary recall metrics (`p15, r15, p40, r40`) are computed explicitly using integer masks against the raw integer class predictions (e.g. `mask_true = np.isin(true_classes, [7])`).
- Midpoint approximations (e.g. `45.0`) were removed entirely from the classification pipeline and were not used in the evaluation metrics.

## CHECK 9 — FINAL SCIENTIFIC INTERPRETATION
**Verdict: GENUINE CLASS COLLAPSE**
There are no evaluation or decoding bugs. The Phase 13B scientific result is mathematically sound and the pipeline executed the experimental intent flawlessly. The frozen representation truly fails to distinguish extreme tall heights, resulting in the model predicting 0 pixels in the >30m and >40m bins across the entire unseen city.

## WHAT IS THE ONE NEXT EXPERIMENT?
If the frozen Depth Anything V2 representation lacks the distinguishing metric-scale features to separate a 25m building from a 60m building, the single most critical hypothesis to test next is whether the encoder *can* learn them if allowed to update. 
**The ONE next experiment: Unfreeze the Depth Anything V2 encoder (using LoRA or partial unfreezing to fit in the 4GB VRAM budget) and observe if metric-scale gradients can break the prediction ceiling.**
