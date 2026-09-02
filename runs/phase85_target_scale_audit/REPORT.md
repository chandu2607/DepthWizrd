# Phase 85 terrain target and spatial-scale audit

## Frozen data and split
- region: uttarakhand
- train_bbox: (1774, 2513, 2286, 3025)
- val_bbox: (5834, 7335, 6346, 7847)
- crop_shape: (512, 512)

## 2. Native DEM validation region audit
- min: 3672.767822265625
- max: 6185.8154296875
- mean: 4946.923171530478
- std: 561.3603070934762
- valid_pixel_count: 262144
- gradient_mean/std/median/p95: 8.151977482934452/4.0778504492209775/7.715059145462901/15.410866767566207
- gradient_pct_above_thresholds: {'1': 99.32708740234375, '5': 76.63917541503906, '10': 29.8065185546875, '25': 0.0885009765625, '50': 0.0, '100': 0.0}

## 3. Exact Phase 84 preprocessing audit
- original DEM resolution: (10.0, 10.0)
- original RGB resolution: (10.0, 10.0)
- crop dimensions: 512x512
- resize operation: NONE (direct crop from aligned common-grid raster)
- resize interpolation: NONE
- final model resolution: 512x512
- meters per pixel: 10.0
- physical width/height of 512x512 crop: 5120.0 m x 5120.0 m
- physical area: 26.2144 km^2
- phase84 target min/max/mean/std: -1327.232177734375/1185.8154296875/-53.0768284695223/561.3603070934762
- phase84 target gradient mean/std/median/p95: 8.151977482934452/4.0778504492209775/7.715059145462901/15.410866767566207

## 4. Local-relief target definition audit
- crop median range: 2488.2607421875 to 5240.126953125
- crop target mean range: -53.07682418823242 to 27.310142517089844
- crop target std range: 158.27220153808594 to 561.3602905273438

## 5. Alternative targets (diagnostic only)
- A_absolute_DEM: min=3672.767822265625, max=6185.8154296875, mean=4946.923171530478, std=561.3603070934762, grad_mean=8.151977482934452, grad_std=4.0778504492209775, grad_p95=15.410866767566207
- B_crop_centered_local_relief: min=-1327.232177734375, max=1185.8154296875, mean=-53.0768284695223, std=561.3603070934762, grad_mean=8.151977482934452, grad_std=4.0778504492209775, grad_p95=15.410866767566207
- C_min_max_normalized_local_relief: min=0.0, max=4.476710796356201, mean=2.269763948027463, std=1.000000029671848, grad_mean=0.014521827821387086, grad_std=0.007264230291547539, grad_p95=0.02745271790882434
- D_z_score_normalized_local_relief: min=-2.269763946533203, max=2.206946611404419, mean=-6.4761496026966014e-09, std=1.000000026571558, grad_mean=0.01452182770073818, grad_std=0.0072642301794327005, grad_p95=0.027452700725339015
- E_DEM_gradient_magnitude: min=0.018760740756988525, max=30.26097297668457, mean=8.151977482667434, std=4.077850449320764, grad_mean=0.7411607647096385, grad_std=0.6425540755858765, grad_p95=1.9991617887862982
- F_DEM_gradient_X: min=-22.5888671875, max=24.06787109375, mean=0.9114610413089395, std=5.187227792474715, grad_mean=0.6743147187677375, grad_std=0.5763883458557658, grad_p95=1.768236275178327
- G_DEM_gradient_Y: min=-28.24365234375, max=26.97607421875, mean=-1.8787295906804502, std=7.198325006050897, grad_mean=0.7654589105551712, grad_std=0.6510188286972518, grad_p95=2.0109815177729695
- H_slope_degrees: min=0.10749099403619766, max=71.71340942382812, mean=36.67133558544586, std=13.418447486166015, grad_mean=2.390626732049559, grad_std=1.9614168845197752, grad_p95=6.070759191869734

## 6. RGB vs DEM scale matching
- RGB luminance gradient mean/std/median/p95: 0.0003798587526425092/0.0009364387728877702/0.0/0.002064161916358129
- scale correlations: {"1x": {"pearson": 0.012940484113995954, "rgb_shape": [512, 512], "target_shape": [512, 512]}, "2x": {"pearson": 0.012709368532897784, "rgb_shape": [256, 256], "target_shape": [256, 256]}, "4x": {"pearson": 0.01249575146550452, "rgb_shape": [128, 128], "target_shape": [128, 128]}, "8x": {"pearson": 0.009515572244542702, "rgb_shape": [64, 64], "target_shape": [64, 64]}}

## 7. Height datum / geographic offset analysis
- crop median distribution mean/std/min/max: 4443.00810546875/792.9158910870456/2488.2607421875/5240.126953125

## 10. Final interpretation
Evidence shows the target keeps substantial terrain variation at the exact 512x512 phase-84 crop scale; the dominant issue is that the terrain signal is regional and coarse relative to a 5.12 km crop, not that the target is absent or broken.

SPATIAL_RESOLUTION_IS_PROBLEMATIC
