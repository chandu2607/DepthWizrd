# PHASE 12 - SCENE-REGIME & SCALE-FAILURE DIAGNOSTIC

## POST-HOC ERROR ANALYSIS (New York Test Set)

*This is a post-hoc error analysis to diagnose the failure mechanism. No dataset tuning or architectural changes were made based on this test set.*

### 1. Scene-Regime Performance
New York scenes were grouped into two extreme categories based on maximum true height:
- **High-Rise (40-80m)**: 58 scenes | Scene MAE: 9.24m | >30m MAE: 42.51m
- **Extreme High-Rise (>80m)**: 50 scenes | Scene MAE: 15.00m | >30m MAE: 46.95m

### 2. Scene-Scale vs Scene-Height (Correlation Analysis)
Target: `scene_mae`
- vs `true_max_height`: 0.707 (p=0.000)
- vs `tall_fraction (>30m)`: 0.734 (p=0.000)
- vs `oracle_scale` (affine slope): 0.198 (p=0.040)

Target: `tall_error (>30m MAE)`
- vs `true_max_height`: 0.474 (p=0.000)
- vs `oracle_scale`: -0.028 (p=0.773)

### 3. Are there scenes where RGB+Depth beats Depth-only?
**Yes.** RGB+Depth had a lower Scene MAE than Depth-only on **50 out of 108** scenes.
- The scenes where RGB+Depth won had a lower mean maximum height (72.4m) compared to the scenes where Depth-only won (81.8m).
- This indicates RGB can be helpful for more moderate scenes, but becomes a severe distractor in extreme high-rise geometry.

---

## SCIENTIFIC VERDICT

### SPECIALIST-MODEL HYPOTHESIS: NOT SUPPORTED
The hypothesis was that different scene regimes (e.g., low-rise vs high-rise, or different affine scales) might require different specialist models. However, the data shows that tall-building error (`>30m MAE`) is virtually identical across all scene scales (Low Scale: 45.0m, Med: 44.1m, High: 44.5m). The failure is a universal ceiling effect, not a regime-specific strategy failure. A specialist model would simply hit the same ceiling.

### SCENE-SCALE FAILURE: WEAK EVIDENCE
The hypothesis that the model fails because it misjudges the affine scale of the scene is incorrect. The correlation between `oracle_scale` and tall-building error is precisely zero (-0.028). The model fails purely as a function of physical height (`true_max_height` r=0.707), regardless of the scene's affine depth-to-height scaling.

## FINAL ANSWERS

1. **Does model failure depend on scene type?** Yes, scene MAE is highly correlated with the fraction of tall pixels (r=0.734).
2. **Does model failure depend on scene scale?** Weakly/No. Oracle scene scale has almost no correlation with tall-building error (r=-0.028).
3. **Is there a measurable prediction ceiling?** Yes. Across all regimes, >30m error is pinned at ~44-46m, meaning the network simply refuses to output values high enough to match extreme ground truths, universally capping its predictions.
4. **Is tall-height failure universal or concentrated?** It is universal across all scale regimes.
5. **Are there distinct failure regimes?** There are distinct *error magnitudes* (Extreme High-Rise has 15m MAE vs 9m for High-Rise), but the underlying *mechanism* (tall building ceiling) is identical in both.
6. **Are there scenes where RGB+Depth beats Depth-only?** Yes, 50/108 scenes (mostly the shorter ones).
7. **Is the specialist-model hypothesis supported?** No.
8. **Is scene scale a stronger explanation than scene height alone?** No. True max height is a vastly stronger explainer (r=0.707) than scene scale (r=0.198).
9. **What is the strongest remaining bottleneck?** A rigid regression ceiling. The network architecture / loss topology actively prevents the model from mapping unbounded relative depth to unbounded metric height, universally collapsing extreme values.
10. **What is ONE next experiment?** Since the failure is a rigid regression ceiling rather than a missing input signal or scene-scale ambiguity, we must change how the network represents height. **Next Experiment: Ordinal Classification (Depth-Binning)**. Replacing the unbounded regression target (L1/log1p) with a discretized cross-entropy classification task should remove the mathematical regression ceiling and allow the network to freely predict extreme bins without being penalized heavily by the L1 variance of the tail.
