# Geospatial Alignment Report

All regions were aligned by reprojecting the DEM onto the optical grid before any crop.
This was done using rasterio.reproject with bilinear interpolation for elevation and the optical B04 tile as the target geospatial grid. The optical CRS and transform were preserved, and the DEM was resampled into the same grid before training.

The resulting arrays are physically matched on a common grid, so each training pixel is tied to the same ground location in RGB and terrain reference.
