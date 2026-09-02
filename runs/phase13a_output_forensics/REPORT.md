# PHASE 13A - OUTPUT-CEILING FORENSICS

*Read-only forensic check of the model architecture, target transforms, and dataset statistics to determine the exact nature of the height prediction ceiling.*

## CHECK 1 — FINAL OUTPUT LAYER
- **Layer**: 
n.Conv2d(w, 1, 1)
- **Activation**: None (Linear combination)
- **Clipping/Bounding**: None
- **Verdict**: **VERIFIED.** The raw model output can theoretically represent arbitrarily large positive or negative values. There is no mathematical architectural bound.

## CHECK 2 — TARGET TRANSFORM
- **Log1p Transform**: 
p.log1p(np.maximum(gt_r, 0.0)) -> Transforms [0, inf) to [0, inf). Does not clip the ceiling.
- **Inverse Transform**: 
p.expm1(pred) -> Computes ^{x} - 1$.
- **Verdict**: **VERIFIED.** No clipping occurs in the preprocessing or postprocessing. A raw network prediction of ~5.2 is sufficient to output >180m.

## CHECK 3 — LOSS FUNCTION
- **Implementation**: Standard unweighted _masked_l1 (Mean Absolute Error).
- **Verdict**: **VERIFIED.** The loss does not mathematically clip or prohibit large outputs. However, it imposes an extreme statistical penalty: because it averages over all pixels, predicting a rare large value (e.g. 50m) incorrectly carries a massive geometric penalty, pulling the network towards the median (conservative) prediction.

## CHECK 4 — DATASET TARGET RANGE (Training Split)
- **Max Height**: 183.17m
- **P99**: 36.51m
- **>30m pixels**: ~6.64M (2.71%)
- **>40m pixels**: ~1.98M (0.81%)
- **>100m pixels**: ~206K (0.08%)
- **Verdict**: **VERIFIED.** The training data does contain extreme heights. The ceiling is not caused by an absolute lack of tall examples.

## CHECK 5 & 6 — PREDICTION VS TARGET RANGE (New York Test)
| Quantity | True target (New York) | Predicted (Depth-Only, Seed 0) |
|---|---:|---:|
| >30m Bin Mean | ~35.5m | 2.98m |
| >40m Bin Mean | ~54.3m | 3.66m |
- **Verdict**: **SUPPORTED.** The model is actively predicting near-ground values (3-4 meters) for pixels that are physically 40-80 meters tall.

## CHECK 8 — TYPE OF CEILING
**Classified as: EMERGENT MODELING CEILING**
There is no hard implementation, target, or pipeline ceiling. The network mathematically *can* output 180m, but it learns not to. Because the model must infer absolute scale from ambiguous relative depth features, it defaults to a highly conservative strategy (predicting near the dataset median) to minimize expected L1 loss. 

## CHECK 10 — RE-EVALUATING ORDINAL CLASSIFICATION
Does ordinal classification actually address this?
**PLAUSIBLE.** Regression (L1/L2) penalizes errors geometrically; a wrong prediction of 50m incurs a huge loss, forcing uncertainty towards the median. Classification (Cross-Entropy) penalizes all incorrect bins equally. This mathematically removes the magnitude-based penalty, freeing the network to predict extreme outlier bins without risking catastrophic loss if it misjudges the scale slightly. It sacrifices continuous precision for tail-end recall.

---

## FINAL ANSWERS

1. **Is there a real mathematical/output ceiling?** No.
2. **Can C_log1p theoretically output >40m, >100m, etc.?** Yes, easily.
3. **Is log1p responsible?** No.
4. **Is inverse log responsible?** No.
5. **Is the loss responsible?** Yes, statistically (but not mathematically). Expected L1 minimization over a highly skewed distribution forces conservative median-seeking.
6. **Is target preprocessing responsible?** No.
7. **Is the ceiling implementation-forced or learned?** It is an **emergent, learned ceiling** driven by optimization dynamics on an ambiguous task.
8. **Does the existing evidence actually justify ordinal classification?** Yes. Since the bottleneck is the expected-value optimization of L1 regression, changing the topology to Classification directly attacks the correct mechanism.
9. **What is the ONE best next experiment?** Switch to **Ordinal Classification (Depth-Binning)**. Discretize the height into bins (e.g., [0-2m], [2-5m] ... [30-40m], [>40m]) and train a cross-entropy classifier to prove whether the model can retrieve the tail when the L1 regression penalty is removed.
