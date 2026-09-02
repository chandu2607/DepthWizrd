# Phase 64 — Data Sources

## Real sources used

- Sentinel-2 L2A public AWS COGs (Uttarakhand, UTM zone 44N)
- Copernicus DEM public AWS COGs (30m, EPSG:4326)

## Access note

The raw Azure blob URLs from the Planetary Computer route were blocked with HTTP 409 / PublicAccessNotPermitted. The working public AWS COG endpoints yielded actual raster bytes that were successfully read by rasterio.
