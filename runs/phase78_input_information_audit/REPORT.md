# Phase 78 input-information audit

## Exact model input path
- Exact Phase 77 code path: TerrainDataset.__init__ creates DepthAnythingV2, calls .infer(...) on the RGB crop, then TerrainDataset.__getitem__ concatenates RGB and depth into a 4-channel feature tensor, which is passed to TerrainHead.forward.
- The actual feature tensor is produced by: np.concatenate([self.rgb, self.depth[None, ...]], axis=0).
- Input feature tensor stats: shape=[4, 512, 512], dtype=torch.float32, min=-1.5172373056411743, max=3.397331476211548, mean=0.03217294067144394, std=0.5033258199691772, finite_count=1048576.

## Depth Anything V2 usage
- DepthAnythingV2 is instantiated in TerrainDataset.__init__ and its output is computed via depth_model.infer(...).
- The result is normalized and included as the 4th feature channel before TerrainHead sees it.
- Therefore, the model is not receiving RGB directly only; it receives concatenated RGB + depth features.

## Input diagnostic summary
- RGB safe stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 0.21130692958831787, "mean": 0.04289725050330162, "min": 0.0, "nan_fraction": 0.0, "shape": [3, 512, 512], "std": 0.06317036598920822}
- Depth safe stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 3.397331476211548, "mean": -2.9802322387695312e-08, "min": -1.5172373056411743, "nan_fraction": 0.0, "shape": [512, 512], "std": 0.9999974966049194}
- Terrain feature safe stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 3.397331476211548, "mean": 0.03217293322086334, "min": -1.5172373056411743, "nan_fraction": 0.0, "shape": [4, 512, 512], "std": 0.5033255815505981}
- RGB luminance/target corr: {"n": 262144, "pearson": 0.5045134533956394, "spearman": 0.5357673765328957}
- Depth/target corr: {"n": 262144, "pearson": 0.5062018506796298, "spearman": 0.426071987128898}
- Feature/target corr: {"n": 262144, "pearson": 0.5045134525975854, "spearman": 0.5357673765328957}

## Alignment
- Crop bbox: [1774, 2513, 2286, 3025]
- RGB shape: [3, 512, 512]
- DEM shape: [512, 512]
- Depth shape: [512, 512]
- Feature tensor shape: [4, 512, 512]
- Resize operations: RGB crop from aligned RGB; DEM crop from aligned DEM; Depth Anything V2 infer with target_hw=(512,512); concatenation into 4x512x512.

## Sensitivity diagnostic
- Original prediction stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 2.5207905769348145, "mean": 0.39259523153305054, "min": -1.805009126663208, "nan_fraction": 0.0, "shape": [512, 512], "std": 0.25813645124435425}
- Shuffled-input prediction stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 2.2819948196411133, "mean": 0.4121682643890381, "min": -0.9581162929534912, "nan_fraction": 0.0, "shape": [512, 512], "std": 0.30890700221061707}
- Constant-input prediction stats: {"dtype": "float32", "finite_fraction": 1.0, "inf_fraction": 0.0, "max": 5.013762950897217, "mean": 0.2912434935569763, "min": -4.237680912017822, "nan_fraction": 0.0, "shape": [512, 512], "std": 0.32031673192977905}
- The model produces spatially varying predictions for the real input, so the issue is not a complete lack of propagation of spatial input.

## Gradient sanity
- Loss: 0.515271
- Gradient norm: 9.297738
- Parameters with nonzero grad: 44
- NaN gradients: 0
- Inf gradients: 0

## Diagnosis
ARCHITECTURE_REPRESENTATION_FAULT
