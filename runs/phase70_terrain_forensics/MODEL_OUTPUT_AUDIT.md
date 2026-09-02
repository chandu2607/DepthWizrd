# Model Output Audit

## Current architecture semantics
The current production-style architecture is in `depthwizard/models/building_conditioned_net.py`.

- `BuildingConditionedHeightNet` uses `SmallFusionUNet(w=w, in_channels=4, out_channels=C_feat + 1)`.
- The output is split into `feat_map` and `mask_logits`.
- `mask_logits` is passed through a sigmoid to form a building mask probability map.
- Connected components are computed on the mask, then geometry/depth features are pooled for each building component.
- The expert heads regress per-component height estimates and are then combined into object-wise predictions.
- The model is therefore object-conditioned, building / nDSM-like, and not a dense terrain DTM regressor.

## Relative-depth prior
`depthwizard/depth/depth_anything.py` returns a dense relative-depth map. That output is scale- and shift-ambiguous and is not a metric terrain-elevation sensor.

## Conclusion
The current output is not a terrain DTM and should not be called a terrain-elevation prediction without an explicit dense terrain regression head.
