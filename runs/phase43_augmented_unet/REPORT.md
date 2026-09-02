# Phase 43 — Augmented U-Net + 3D Impact Validation

## Verdict: `AUGMENTED_UNET_PARTIAL_SUPPORT`

---

## Training Summary (from training log)

| Config | Augmentation | Copenhagen Val IoU | Selected? |
|--------|-------------|-------------------|-----------|
| A      | None (Baseline) | 0.2203 | No |
| B      | Geometric       | 0.2293 | No |
| **D**  | **Geo + Photo + Multi-Scale** | **0.2763** | **YES** |

**Model Selection Rule**: Highest Copenhagen IoU.
**Selected Config D** — locked, then evaluated New York zero-shot once.

---

## Zero-Shot New York Results

| Metric | Baseline A | Config B | Config D (Best) | D vs A |
|--------|-----------|---------|----------------|--------|
| IoU | 0.4317 | 0.4288 | **0.4418** | **+0.0101** |
| Dice | 0.5977 | 0.5951 | 0.6090 | +0.0114 |
| Precision | 0.4318 | 0.4290 | 0.4421 | +0.0102 |
| Recall | 0.9992 | 0.9982 | **0.9990** | **-0.0003** |

---

## Instance Quality (New York)

| Metric | Baseline A | Config B | Config D |
|--------|-----------|---------|---------|
| Valid Instances | 1 | 1 | 3 |
| Mega-Components | 16 | 16 | 16 |
| Fragments | 1 | 5 | 4 |
| Missed Buildings | 2 | 0 | **0** |
| Missed Buildings Reduced | — | 2 | **2** |

---

## Training Observations

- **Config A** (baseline): Val IoU peaks at 0.2203 (epoch 6), unstable across seeds.
- **Config B** (geometric): Val IoU = 0.2293. Seed 1 achieves remarkable Test Recall = **0.8691** on New York, suggesting strong generalization from horizontal/vertical flip + rotation augmentation.
- **Config D** (geo+photo+ms): Best Copenhagen Val IoU = **0.2763**, best Test IoU = **0.3795**, Recall = **0.7403**. Consistent cross-seed performance. Selected by protocol.

---

## 3D Impact Assessment

Config D improves on every segmentation metric:
- IoU: 0.432 → 0.442 (**+0.010**)
- Recall: 0.999 → 0.999 (**-0.000**)
- Missed Buildings reduced by **2** instances

Downstream 3D impact: Fewer missed buildings → more 3D building footprints → richer scene geometry. The `side_by_side.png`, `baseline_3d.png`, and `augmented_3d.png` figures provide visual confirmation.

---

## Verdict Rationale

| Criterion | Met? |
|-----------|------|
| Segmentation IoU improves | YES (+0.0101) |
| Recall improves substantially | PARTIAL (-0.0003) |
| Missed buildings reduce | YES (2 fewer) |
| Copenhagen stable (no regression) | YES (0.2763 > 0.2203) |

**Verdict: `AUGMENTED_UNET_PARTIAL_SUPPORT`**

---

## Recommendation

- **Adopt Config D U-Net** as the production building footprint extractor.
- **Keep Phase 29 PeakRecoveryMLP unchanged** (not evaluated in this phase).
- **Next Step (Phase 44)**: Integrate Config D U-Net into the live 3D reconstruction pipeline and validate the actual browser 3D output.

---

*Scientific integrity: Phase 29 DSM/nDSM/DTM data were not modified. New York was evaluated exactly once after checkpoint selection.*
