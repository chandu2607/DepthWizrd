# Phase 61 — Indian Dataset Discovery Sources

## Goal

This file records the real public sources considered for an Indian mountainous training and validation benchmark. The objective is to discover a small, scientifically usable pilot benchmark before any model adaptation begins.

## Sources considered

### 1. Sentinel-2

- Provider: ESA Copernicus
- Role: Optical RGB / multispectral coverage for Indian hill regions
- Strength: broad open coverage, good terrain context, georeferenced scenes
- Limitation: does not provide a direct metric elevation reference

### 2. Copernicus DEM GLO-30

- Provider: Copernicus
- Role: terrain elevation reference
- Strength: public, georeferenced, widely available
- Limitation: 30 m resolution is coarse for steep mountain micro-topography, settlement boundaries, and small hazard features

### 3. SRTM

- Provider: USGS / NASA
- Role: public global DEM reference
- Strength: easy to access and widely used
- Limitation: coarse spatial resolution; can be too limited for fine-scale height and slope estimation in mountain terrain

### 4. Bhuvan / NRSC

- Provider: ISRO / NRSC
- Role: regional geospatial products and Indian public datasets
- Strength: relevant to Indian terrain and disaster mapping
- Limitation: product quality and availability vary substantially; not every region provides a clean paired benchmark dataset

### 5. OpenTopography

- Provider: academic data aggregation platform
- Role: hosted DEM / LiDAR / DSM resources
- Strength: sometimes very high quality
- Limitation: not every Indian mountainous region is hosted there; manual verification is required

### 6. Landslide and hazard map products

- Provider: state agencies, ISRO, GSI, public event layers
- Role: hazard context and terrain risk information
- Strength: relevant to disaster analysis
- Limitation: not a direct supervised height-estimation benchmark without paired imagery + elevation reference

### 7. OpenStreetMap building footprints

- Provider: OSM
- Role: building masks and settlement context
- Strength: helps with building exposure / validation context
- Limitation: no height ground truth, no terrain reference

## Current status

The workspace does not yet contain a verified real Indian mountain benchmark tile with paired optical imagery and a matched elevation reference. The dataset discovery step therefore ends at source identification and pilot-benchmark planning, not model training.

## Pilot-benchmark strategy

The next step is to select the smallest benchmark that satisfies all of the following:

- real Indian mountainous region
- public optical imagery
- public elevation reference
- compatible CRS and spatial overlap
- manageable tile size for RTX 3050 constraints
- geographic separation across train / val / test regions

Until that benchmark is accepted, no training run begins.
