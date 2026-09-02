import json
import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
import requests

OUT = Path(__file__).resolve().parent
ORIG = OUT / 'ORIGINAL_DATA'
ALIGNED = OUT / 'ALIGNED_DATA'
for d in [ORIG, ALIGNED]:
    d.mkdir(exist_ok=True, parents=True)

# Candidate real Uttarakhand pair discovered from public STAC catalogs
s2_item_id = 'S2B_MSIL2A_20241231T052119_R062_T44RMV_20241231T073111'
dem_item_id = 'Copernicus_DSM_COG_10_N31_00_E080_00_DEM'

s2_item_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/' + s2_item_id
r_s2 = requests.get(s2_item_url, timeout=60)
print('S2 item status:', r_s2.status_code)
s2_item = r_s2.json()
print('S2 bbox:', s2_item.get('bbox'))
print('S2 asset keys:', sorted(s2_item.get('assets', {}).keys())[:20])

# pick a small real optical crop using B04, B03, B02 and a DEM tile data asset
s2_band_urls = {
    'B02': s2_item['assets']['B02']['href'],
    'B03': s2_item['assets']['B03']['href'],
    'B04': s2_item['assets']['B04']['href'],
}
print('B04 href sample:', s2_band_urls['B04'][:200])

# DEM item metadata
url_dem = 'https://planetarycomputer.microsoft.com/api/stac/v1/collections/cop-dem-glo-30/items/' + dem_item_id
r_dem = requests.get(url_dem, timeout=60)
print('DEM item status:', r_dem.status_code)
dem_item = r_dem.json()
print('DEM bbox:', dem_item.get('bbox'))
print('DEM asset keys:', list(dem_item.get('assets', {}).keys()))
dem_href = dem_item['assets']['data']['href']
print('DEM href sample:', dem_href[:200])

# Attempt to read the real remote raster metadata and a small crop window.
# For S2, use UTM zone 44N. A modest 2048x2048 crop near the mountainous overlap region.
for name, href in [('B04', s2_band_urls['B04']), ('DEM', dem_href)]:
    print('\nOpening', name, href[:120])
    try:
        with rasterio.open(href) as src:
            print('  profile:', src.profile)
            print('  width/height:', src.width, src.height)
            print('  crs:', src.crs)
            print('  bounds:', src.bounds)
            print('  transform:', src.transform)
            if name == 'B04':
                # crop a moderate tile around the central mountainous overlap
                win = Window(1500, 1200, 2048, 2048)
                arr = src.read(1, window=win)
                out = ORIG / 'uttarakhand_s2_b04_crop.tif'
                profile = src.profile.copy()
                profile.update(width=arr.shape[1], height=arr.shape[0], transform=src.window_transform(win))
                with rasterio.open(out, 'w', **profile) as dst:
                    dst.write(arr, 1)
                print('  wrote B04 crop to', out)
            else:
                win = Window(500, 500, 2048, 2048)
                arr = src.read(1, window=win)
                out = ORIG / 'uttarakhand_dem_crop.tif'
                profile = src.profile.copy()
                profile.update(width=arr.shape[1], height=arr.shape[0], transform=src.window_transform(win))
                with rasterio.open(out, 'w', **profile) as dst:
                    dst.write(arr, 1)
                print('  wrote DEM crop to', out)
    except Exception as e:
        print('  ERROR', type(e).__name__, e)
