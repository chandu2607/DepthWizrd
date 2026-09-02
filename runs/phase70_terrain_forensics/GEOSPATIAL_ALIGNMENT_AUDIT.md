# Geospatial Alignment Audit

This audit checks whether the raw RGB and the raw DEM arrays are spatially aligned. The phase69 pilot did not reproject the DEM before center-cropping, so raw array indices are not guaranteed to refer to the same physical area.
## uttarakhand
- RGB CRS: EPSG:32644
- RGB transform: | 10.00, 0.00, 399960.00|
| 0.00,-10.00, 3400020.00|
| 0.00, 0.00, 1.00|
- RGB shape: 10980 x 10980
- RGB bounds: BoundingBox(left=399960.0, bottom=3290220.0, right=509760.0, top=3400020.0)
- DEM CRS: EPSG:4326
- DEM transform: | 0.00, 0.00, 80.00|
| 0.00,-0.00, 31.00|
| 0.00, 0.00, 1.00|
- DEM shape: 3600 x 3600
- DEM bounds: BoundingBox(left=79.99986111111112, bottom=30.000138888888888, right=80.99986111111112, top=31.000138888888888)
- DEM bounds reprojected to RGB CRS: (403536.58523938234, 3318800.7426035865, 499986.7408441257, 3430046.5614345926)
- Overlap in RGB CRS: (403536.58523938234, 3318800.7426035865, 499986.7408441257, 3400020.0)
- Sampled pixel-to-coordinate checkpoints:
  - RGB pixel (0, 0) -> world (399960.000000, 3400020.000000) in EPSG:32644
  - RGB pixel (100, 100) -> world (400960.000000, 3399020.000000) in EPSG:32644
  - RGB pixel (512, 512) -> world (405080.000000, 3394900.000000) in EPSG:32644
  - RGB pixel (5490, 5490) -> world (454860.000000, 3345120.000000) in EPSG:32644
  - RGB pixel (10979, 10979) -> world (509750.000000, 3290230.000000) in EPSG:32644

## himachal
- RGB CRS: EPSG:32643
- RGB transform: | 10.00, 0.00, 499980.00|
| 0.00,-10.00, 3500040.00|
| 0.00, 0.00, 1.00|
- RGB shape: 10980 x 10980
- RGB bounds: BoundingBox(left=499980.0, bottom=3390240.0, right=609780.0, top=3500040.0)
- DEM CRS: EPSG:4326
- DEM transform: | 0.00, 0.00, 76.00|
| 0.00,-0.00, 31.00|
| 0.00, 0.00, 1.00|
- DEM shape: 3600 x 3600
- DEM bounds: BoundingBox(left=75.99986111111112, bottom=30.000138888888888, right=76.99986111111112, top=31.000138888888888)
- DEM bounds reprojected to RGB CRS: (595454.9551666201, 3319221.4970407668, 692901.4367747928, 3431334.001799481)
- Overlap in RGB CRS: (595454.9551666201, 3390240.0, 609780.0, 3431334.001799481)
- Sampled pixel-to-coordinate checkpoints:
  - RGB pixel (0, 0) -> world (499980.000000, 3500040.000000) in EPSG:32643
  - RGB pixel (100, 100) -> world (500980.000000, 3499040.000000) in EPSG:32643
  - RGB pixel (512, 512) -> world (505100.000000, 3494920.000000) in EPSG:32643
  - RGB pixel (5490, 5490) -> world (554880.000000, 3445140.000000) in EPSG:32643
  - RGB pixel (10979, 10979) -> world (609770.000000, 3390250.000000) in EPSG:32643

## sikkim
- RGB CRS: EPSG:32645
- RGB transform: | 10.00, 0.00, 499980.00|
| 0.00,-10.00, 3100020.00|
| 0.00, 0.00, 1.00|
- RGB shape: 10980 x 10980
- RGB bounds: BoundingBox(left=499980.0, bottom=2990220.0, right=609780.0, top=3100020.0)
- DEM CRS: EPSG:4326
- DEM transform: | 0.00, 0.00, 88.00|
| 0.00,-0.00, 28.00|
| 0.00, 0.00, 1.00|
- DEM shape: 3600 x 3600
- DEM bounds: BoundingBox(left=87.99986111111112, bottom=27.000138888888888, right=88.99986111111112, top=28.000138888888888)
- DEM bounds reprojected to RGB CRS: (598311.5518962743, 2986843.6280741184, 698440.2054054897, 3098829.4203634355)
- Overlap in RGB CRS: (598311.5518962743, 2990220.0, 609780.0, 3098829.4203634355)
- Sampled pixel-to-coordinate checkpoints:
  - RGB pixel (0, 0) -> world (499980.000000, 3100020.000000) in EPSG:32645
  - RGB pixel (100, 100) -> world (500980.000000, 3099020.000000) in EPSG:32645
  - RGB pixel (512, 512) -> world (505100.000000, 3094900.000000) in EPSG:32645
  - RGB pixel (5490, 5490) -> world (554880.000000, 3045120.000000) in EPSG:32645
  - RGB pixel (10979, 10979) -> world (609770.000000, 2990230.000000) in EPSG:32645

