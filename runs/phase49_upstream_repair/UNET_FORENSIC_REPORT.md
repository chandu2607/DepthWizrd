# U-Net Forensic Report

## Checkpoint
- Path: c:\Users\chand\OneDrive\Desktop\DepthWizard\runs\phase43_augmented_unet\unet_config_D.pt
- SHA256: 93ebf2f2da89f57125ba954ad1dfdc7c9abbf3f8330f62bff5d1f2edd3d7ea1e
- Class: BuildingConditionedEstimator -> BuildingConditionedHeightNet -> SmallFusionUNet
- Parameter count: 479281
- State dict keys: 106
- Missing keys: []
- Unexpected keys: []
- Output: one segmentation logit channel, raw shape [1, 1, 256, 256]

## Training vs production preprocessing

| parameter | training | production | match? |
| --- | --- | --- | --- |
| image_shape | source tile 512x512 -> model 256x256 | NYC source 512x512 -> model 256x256 | Yes |
| color_order | RGB (stored as np.uint8 RGB before /255.0, in _prep_x) | RGB (raster loader returns RGB) | Yes |
| pixel_scale | uint8 0..255 => /255 if max>1.5 | uint8 0..255 => app keeps uint8, passed to estimator._prep_x | Yes, same conversion path |
| depth_normalization | depth_norm = (depth - d_mean) / (d_std + 1e-6) | same function _prep_x applies same normalization | Yes |
| resize | cv2.resize(rgb, (res,res)); cv2.resize(depth, (res,res)) | same resize in estimator._prep_x | Yes |
| crop | none in _prep_x; whole tile at train_res | none in _prep_x; production uses full tile then resize | Yes |
| channels | x = concat([rgb.transpose(2,0,1), depth_norm[None]], axis=0) => 4 channels | same concat in _prep_x | Yes |
| model_output | mask_logits, then sigmoid | same direct sigmoid in model.forward | Yes |

## Raw logits stats
- min: 0.06895217299461365
- max: 0.8311086893081665
- mean: 0.5111035108566284
- median: 0.5272794961929321
- p1: 0.2344927132129669
- p5: 0.31279023736715317
- p25: 0.42319855839014053
- p50: 0.5272794961929321
- p75: 0.6006623208522797
- p95: 0.6793302297592163
- p99: 0.7662575334310532

## Probability stats
- min: 0.5172312259674072
- max: 0.6965893507003784
- mean: 0.6246652603149414
- median: 0.6288484334945679
- p1: 0.5583560347557068
- p5: 0.5775661766529083
- p25: 0.6042483896017075
- p50: 0.6288484036922455
- p75: 0.6458078175783157
- p95: 0.6635891944169998
- p99: 0.682710799574852

## Threshold fractions
- > 0.1: 1.0
- > 0.2: 1.0
- > 0.3: 1.0
- > 0.4: 1.0
- > 0.5: 1.0
- > 0.6: 0.78759765625
- > 0.7: 0.0
- > 0.8: 0.0
- > 0.9: 0.0

## Model collapse check
- probability_std: 0.027777891606092453
- probability_iqr: 0.04155939817428589
- probability_entropy: 0.6600914001464844
- collapsed: True


## Diagnosis

**U_NET_MODEL_COLLAPSE**

Config D loads cleanly and emits one segmentation logit channel. Training and production preprocessing match: RGB input, 0..1 pixel scaling, checkpoint-carried depth normalization buffers, bilinear resize to 256x256, and four input channels. However, the NYC probability field has 100% of pixels above 0.5 and the training-distribution control has 99.913% above 0.5. The model is therefore collapsed/elevated even on same-distribution imagery; this is not explained by NYC-only domain shift or threshold selection. The Phase 29 baseline is also elevated on NYC, but its direct comparison does not change the Config D diagnosis. No production changes or threshold tuning were performed.
