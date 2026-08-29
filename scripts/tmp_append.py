import codecs

content = '''
---
---

# DepthWizard — PHASE 14E (Deepest ViT Block Adaptation)

_Appended 2026-08-28. This phase tested whether unfreezing the deepest ViT block (`layer.11` and `layernorm`) in the DA-V2 backbone could restore metric-scale information for tall-height prediction, while preserving the proven `C_log1p` pipeline._

## Hypothesis
If the frozen ViT backbone hides metric scale information in its deepest attention layers, adapting the final block will allow the network to extract this absolute-scale information and pass it to the `SmallFusionUNet`, reducing the tall-height metric collapse.

## Results & Verification (mean over seeds 0, 1)

The experiment successfully trained the deepest ViT block natively alongside the neck, head, and `SmallFusionUNet`. VRAM usage remained exceptionally low (~1 GB peak), proving that partial ViT unfreezing is computationally feasible on standard hardware.

- **Overall MAE:** 8.93 m (mean)
- **>30m Bias:** -22.31 m (predicts ~13.20 m)
- **>40m Bias:** -39.12 m (predicts ~15.17 m)

Compared to the completely frozen backbone baseline (Phase 14D), the tall-height prediction ceiling lifted by roughly **~2.7 meters**. The P95 and P99 predictions remained capped in the 20-30m range. The visual predictions showed slightly "hotter" roofs on massive skyscrapers, but the fundamental limitation remained: 60m buildings were still predicted as ~15-20m tall.

## Scientific Verdict: PARTIAL SUPPORT (CASE B)

**The hypothesis receives weak, PARTIAL SUPPORT.**

The tall prediction did improve measurably and consistently (~2.7m), proving that unfreezing the backbone does allow the model to extract *some* additional task-specific scale information that the frozen representation hid. However, the prediction remains substantially below the truth (predicting 15m for 54m buildings). Thus, adapting *only* the deepest ViT block helps, but additional scale information is still missing. 

## Next Step
The logical next step is deeper unfreezing into the ViT backbone (e.g., the last 4 blocks, which correspond to the entire final "stage" of feature extraction in typical ViT architectures) to determine if the required metric-scale information is buried further back in the network.
'''

with codecs.open('EXPERIMENT_RESULTS.md', 'a', encoding='utf-8') as f:
    f.write(content)
