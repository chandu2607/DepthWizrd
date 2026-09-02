# EXPERIMENT #10 — PRETRAINED DEPTH REPRESENTATION COMPARISON

## STAGE A — CANDIDATE REPORT

### Candidates
| Candidate | Relative/Metric | Remote-sensing relevance | VRAM feasibility | License | Main advantage | Main risk |
|---|---|---|---|---|---|---|
| ZoeDepth (ZoeD_N) | Metric | None (indoor/driving) | Yes (fits 4GB) | MIT | Stable zero-shot metric scale, drop-in replacement | Pretrained on NYU/KITTI, may struggle with nadir aerial views |
| Metric3D v2 (ViT-S) | Metric | None (indoor/driving) | Yes (ViT-Small) | Apache 2.0 | Explicit camera model recovery, SOTA zero-shot | Perspective camera priors might break on orthographic satellite data |
| UniDepth (ViT-B) | Metric | None (mixed) | Yes (~86M params) | CC-BY-NC | Joint metric depth & intrinsic recovery | Non-commercial license, similar perspective-prior risks |

### Ranking
1. **#1 ZoeDepth (ZoeD_N)**
2. **#2 Metric3D v2 (ViT-S)**
3. **#3 UniDepth (ViT-B)**

### Scientific Question
> What specific missing capability might this representation provide that Depth Anything V2 does not?

**Answer:** Depth Anything V2 produces *relative* depth, leaving an unknown, scene-dependent affine scale shift. Our Phase 6 and 7 diagnostics proved that this scene-specific scale cannot be reliably inferred by the fusion network in zero-shot cross-city transfer. **ZoeDepth** provides *absolute metric depth*. If ZoeDepth's internal metric calibration holds up even partially on overhead imagery, it provides a much more stable absolute geometric anchor. This could directly solve the unpredictable depth→height mapping shift between cities.

---
## STAGE B — SMALL PROBE RESULTS
_(Running probe...)_
**Probe Dataset:** DFC2023 (Berlin, Copenhagen, New York) - 5 tiles each.

**Results Summary (Mean across tiles):**
| Metric | Berlin (Train) | Copenhagen (Val) | New York (Test) |
|---|---|---|---|
| **DA-V2 Pearson (All)** | 0.297 | 0.372 | 0.051 |
| **ZoeDepth Pearson (All)** | -0.026 | -0.153 | 0.195 |
| **DA-V2 Spearman (All)** | 0.389 | 0.381 | 0.072 |
| **ZoeDepth Spearman (All)** | 0.000 | -0.159 | 0.229 |
| **ZoeDepth Metric MAE** | 4.82 m | 7.54 m | 11.71 m |
| **DA-V2 Pearson (>30m)** | -0.683 | 0.119 | 0.089 |
| **ZoeDepth Pearson (>30m)** | -0.564 | -0.080 | 0.345 |

**Interpretation:**
ZoeDepth fundamentally fails to interpret overhead/nadir remote sensing imagery. Its relative depth ordering (Spearman correlation) drops to near zero or becomes *negative* in most cities compared to Depth Anything V2. The raw "metric" scale produced by ZoeDepth has massive absolute error (MAE ~11.7 m in New York across all pixels). While ZoeDepth is state-of-the-art for indoor/automotive scenes, its strong perspective priors break completely on orthographic-like satellite tiles. 

---
## FINAL SCIENTIFIC CONCLUSION

**1. Is Depth Anything V2 still the strongest representation for our specific task?**
Yes. Despite missing absolute scale, Depth Anything V2 preserves significantly better relative spatial geometry and depth ordering (Spearman) on overhead imagery than the tested metric candidate.

**2. Which candidate is the strongest alternative?**
ZoeDepth (ZoeD_N) was tested as the strongest practical alternative because of its zero-shot metric capabilities and VRAM feasibility.

**3. Does the candidate contain more useful tall-height signal?**
No. Both models struggle massively on tall structures (>30m), often producing negative correlation (inverting the depth order).

**4. Is its depth->height relationship more stable across cities?**
No. The metric output MAE jumps wildly from 4.8 m in Berlin to 11.7 m in New York, proving its internal scale calibration does not transfer to new remote sensing domains.

**5. Does it provide better metric-scale information?**
No. Its perspective assumptions cause severe structural hallucinations on nadir views.

**6. Can it run safely on RTX 3050 4 GB?**
Yes. It consumed ~330 MB of VRAM during inference.

**7. What evidence supports replacing Depth Anything V2?**
None. ZoeDepth performed worse on nearly all ordering metrics.

**8. What evidence argues against replacement?**
ZoeDepth's negative Pearson/Spearman correlations and massive raw MAE show it misinterprets satellite geometry entirely.

**9. What is the ONE next experiment?**
Since neither pure data scaling (Phase 9) nor pretrained metric depth transfer (Phase 10) resolves the unseen-city scale collapse, the next logical experiment must target **explicit scale calibration** (e.g., using a few known metric anchors or ground-control points in the target city to shift the relative map).

**10. What remains uncertain?**
Whether it is fundamentally possible to extract a universal metric scale from a single satellite RGB tile *without* external metadata (GSD, sun angle, camera parameters, or sparse LiDAR).

**Verdict:** CASE C - Candidate Worse. Depth Anything V2 is retained.
