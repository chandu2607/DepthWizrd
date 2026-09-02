# Phase 63 — Source Access Audit

## Purpose

This report records the current access status of the candidate public sources for a small real Indian mountainous benchmark. The goal is to confirm actual current access and the realistic acquisition route before any benchmark is accepted.

## 1. Sentinel-2

- Provider: ESA Copernicus
- Access: Public through Copernicus Data Space and browser / API access
- Suitability: Good optical source for a mountain scene
- Limitation: not a direct elevation dataset; must be paired with a DEM, DSM, or DTM from another source
- Current status: source is publicly available and realistic for a small pilot region

## 2. Copernicus DEM GLO-30

- Provider: ESA / Copernicus
- Access: Public through Copernicus Data Space and associated access channels
- Suitability: Strong terrain reference for a coarse DEM baseline
- Limitation: ~30 m resolution is too coarse for fine slope / building-scale analysis, but usable as a terrain reference
- Current status: source is valid for a pilot reference, subject to tile selection and alignment

## 3. Bhuvan / NRSC / ISRO

- Provider: ISRO / NRSC
- Access: Portal and public geospatial product mechanisms; some products may require product-specific access or registration
- Suitability: Strongest Indian source for local terrain and imagery products
- Limitation: access and product pairing are not guaranteed; must be checked case-by-case
- Current status: promising but not yet accepted as a benchmark without direct acquisition and alignment

## 4. CartoDEM / DEM products through Indian public channels

- Provider: Indian geospatial agencies / public channels
- Access: varies by product and portal
- Suitability: useful if a local DEM or orthophoto product is available for the exact selected area
- Limitation: not all products are easy to pair with optical imagery without a quality check
- Current status: candidate only

## 5. SRTM

- Provider: USGS / NASA
- Access: public and widely available
- Suitability: fallback terrain reference only
- Limitation: too coarse for detailed mountain benchmark work
- Current status: useful only if no better pilot reference is available

## Conclusion

The real public sources exist and are usable in principle, but no actual paired Indian optical + elevation benchmark has yet been acquired and aligned in this workspace. This phase therefore does not claim benchmark validity yet.
