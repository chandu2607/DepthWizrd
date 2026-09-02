# Phase 64 Report

## Outcome
A real public Uttarakhand Sentinel-2 band and a real Copernicus DEM tile were successfully downloaded from AWS public COG endpoints and read with rasterio.
The raw Azure blob route was blocked with HTTP 409 / PublicAccessNotPermitted, so the AWS public COG route was used instead.
The pair was aligned by creating a derived crop in UTM 44N and a reprojected DEM crop.
The current production DepthAnythingV2 model was run on the real optical crop without modification.
No training step was started, and no architecture changes were made.
