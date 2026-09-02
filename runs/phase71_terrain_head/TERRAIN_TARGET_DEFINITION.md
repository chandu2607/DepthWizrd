# Terrain Target Definition

Target name: terrain elevation (DEM aligned to optical grid)
Target type: bare-earth terrain reference, treated as continuous elevation map
Target units: meters
Target grid: optical Sentinel-2 B04 grid in UTM per region
Interpolation: bilinear for DEM resampling, nearest-neighbor for any categorical masks
Nodata: NaN retained and excluded from loss and metrics
