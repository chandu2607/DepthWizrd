# Phase 66 — Source Access Audit

## Verified historical evidence

### Region A: Uttarakhand (verified)
- Optical source: Sentinel-2 L2A AWS public COG
- Elevation source: Copernicus DEM GLO-30 AWS public COG
- Verification status: downloaded, opened with rasterio, pixels read successfully
- Evidence: Phase 64 downloaded actual bytes and recorded SHA256 checksums
- Limitation: this DEM is too coarse for high-resolution building-height validation; it is acceptable as terrain context only

### Region B: Himachal Pradesh (candidate, not yet accepted)
- Optical source: publicly indexed Sentinel-2 scenes are expected to exist through Copernicus / STAC access
- Elevation source: public DEM collections and regional products may exist, but no real paired raster was downloaded and read in this workspace
- Verification status: not accepted as a benchmark yet
- Limitation: no actual files, no pixel-read validation, no alignment proof

### Region C: Sikkim (candidate, not yet accepted)
- Optical source: publicly indexed Sentinel-2 scenes are expected to exist through Copernicus / STAC access
- Elevation source: public DEM collections may exist, but no real paired raster was downloaded and read in this workspace
- Verification status: not accepted as a benchmark yet
- Limitation: no actual files, no pixel-read validation, no alignment proof

## Ground-truth hierarchy used

- Level 1: LiDAR / high-resolution photogrammetric DSM — not currently available for the selected Indian benchmark regions in this workspace
- Level 2: high-resolution DSM / DTM — not yet acquired for Himachal or Sikkim
- Level 3: medium-resolution DEM — yes, Uttarakhand has a public Copernicus DEM reference, but it is a terrain reference only
- Level 4: coarse DEM — used only for terrain context, not for building-height ground truth

## Building reference status

- Building footprints: not verified in any accepted Indian region
- Building height labels: not verified in any accepted Indian region
- DSM-derived building heights: not verified in any accepted Indian region

## Overall conclusion

The project has one real terrain benchmark and no accepted high-resolution building ground truth. The benchmark is therefore not yet training-ready for building-specific work and must remain on the evidence gate until the reference data is physically acquired and aligned.
