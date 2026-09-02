# Phase 62 — Public Indian EO and Elevation Sources

## Verified sources

The following sources are the most relevant realistic access paths for a small Indian mountainous benchmark:

### 1. Bhuvan / NRSC / ISRO
- Strong candidate for Indian public geospatial products.
- Relevant because it is explicitly Indian and includes terrain / imagery products.
- Access model: public portal and possibly registration-based download depending on product.
- Limitation: product availability and exact benchmark pairing vary by region and product.

### 2. Copernicus Data Space
- Source for Sentinel-2 optical imagery and Copernicus DEM.
- Public open access is available through the Copernicus ecosystem.
- Limitation: the DEM is only 30 m resolution and not guaranteed to match fine terrain or building height detail.

### 3. SRTM
- Globally available and open.
- Useful fallback terrain reference but coarse.
- Limitation: too coarse for detailed slope and building-level analysis.

### 4. OpenTopography
- Potentially highest-quality reference if a suitable Indian LiDAR / DSM dataset is hosted.
- Limitation: site-specific and not consistently available for Indian hilly terrain.

## Candidate region shortlist

- Uttarakhand
- Himachal Pradesh
- Sikkim
- Jammu & Kashmir / Ladakh
- Arunachal Pradesh

## Current verification status

The sources are real and publicly accessible in principle, but no valid paired optical + elevation benchmark has yet been downloaded and aligned in this workspace. We therefore stop at source validation and benchmark preparation, not a training-ready result.
