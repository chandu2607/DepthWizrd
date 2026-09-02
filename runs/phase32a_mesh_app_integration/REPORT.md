# Phase 32A — Mesh App Integration Report

## Integration Target
Replace `StructuredGrid.extract_surface()` in `app.py` with Phase 31D
edge-aware quad filter (`build_edge_aware_mesh`, dZ_threshold=10.0m).

## DSM Integrity
| Stat | Before | After | Match |
|------|--------|-------|-------|
| min  | 53.0781m | 53.0781m | OK |
| max  | 166.5718m | 166.5718m | OK |
| mean | 86.7612m | 86.7612m | OK |

## Mesh Topology
| Metric | Old (StructuredGrid) | New (Edge-Aware) |
|--------|---------------------|-----------------|
| n_cells | 261121 | 257893 |
| Quads removed | 0 | 3228 (1.24%) |
| Max cell dZ | 35.2m | 8.94m |
| Build time | 0.04s | 0.06s |

## Curtain Artifact Reduction
- Quads with dZ>5m before: 4067
- Quads with dZ>5m after:  839
- Reduction: 79%

## Performance
| Step | Time |
|------|------|
| Mesh build (old) | 0.04s |
| Mesh build (new) | 0.06s |
| Render (elevation) | 2.26s |
| Render (RGB) | 1.10s |

## Success Criteria
| Criterion | Result |
|-----------|--------|
| curtain_artifacts_reduced | PASS |
| building_roofs_preserved | PASS |
| dsm_integrity_ok | PASS |
| texture_alignment_ok | PASS |
| demo_scene_ok | PASS |
| export_ready | PASS |


## Verdict
**`APP_MESH_INTEGRATION_SUCCESS`**

## Next Action
`SHIP_TO_DEMO`
