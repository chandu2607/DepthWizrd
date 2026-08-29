# 🌐 DepthWizard — Absolute 3D Elevation Reconstruction

> **SIH 2024 Demonstration Project**  
> Monocular RGB → Absolute Digital Surface Model (DSM) → Interactive 3D Scene

---

## 📸 What It Does

DepthWizard turns an ordinary satellite RGB image into a fully georeferenced, metric-scale 3D terrain model using:

1. **Depth Anything V2** — monocular relative depth estimation
2. **Phase 29 PeakRecoveryMLP** — building peak height recovery
3. **Phase 30 DTM integration** — absolute terrain ground extraction
4. **Phase 31 PyVista mesh pipeline** — interactive 3D scene generation
5. **Phase 32 Streamlit MVP** — one-page demo dashboard

---

## 🚀 Quick Start — Running the Demo Locally

### Prerequisites
- Python **3.11** (not 3.12 or 3.13 — pyvista/rasterio wheels require 3.11)
- Git
- ~4 GB free disk space

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/chandu2607/DepthWizrd.git
cd DepthWizrd
```

---

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows users:** If `rasterio` fails, install the pre-built wheel:
> ```bash
> pip install rasterio --find-links https://girder.github.io/large_image_wheels
> ```
> Or download from: https://github.com/cgohlke/geospatial-wheels/releases

---

### Step 3 — Download the Demo Data

The satellite GeoTIFF tiles and model checkpoints are **not stored in Git** (too large).  
Ask the project owner (Chandu) to share the `data/` and `runs/` folders via Google Drive or a USB drive.

**Required folder structure after copying:**
```
DepthWizrd/
├── data/
│   └── dfc2023_multicity/
│       ├── rgb/
│       │   ├── SV_NewYork_40.7401_-73.9915.tif   ← demo tile
│       │   ├── SV_NewYork_40.7372_-73.9901.tif
│       │   └── SV_NewYork_40.7373_-74.0034.tif
│       └── dsm/
│           ├── SV_NewYork_40.7401_-73.9915.tif
│           └── ...
└── runs/
    └── phase29_peak_recovery/
        ├── seed_0/model.pt                        ← MLP checkpoint
        └── normalization_stats.json
```

---

### Step 4 — Launch the App

```bash
python -m streamlit run app.py
```

The app will open automatically at: **http://localhost:8501**

---

### Step 5 — Run the Demo

Once the app is open in your browser:

| Step | Action |
|------|--------|
| 1 | Click **🏙️ Load Demo Scene (NYC)** in the left sidebar |
| 2 | Verify the satellite image loads in **Section 1** |
| 3 | Click **🚀 RUN DEPTHWIZARD** |
| 4 | Wait ~10–20 seconds for depth inference + DSM reconstruction |
| 5 | See the **3D rendered mesh** in Section 5 |
| 6 | Change **Camera Angle** / **Render Mode** in the sidebar to explore |
| 7 | Download outputs from **Section 6** (GeoTIFF, nDSM, .vtp mesh) |

---

## 📂 Repository Structure

```
DepthWizrd/
├── app.py                          ← Streamlit demo dashboard (Phase 32)
├── requirements.txt
├── depthwizard/                    ← Core library
│   ├── config.py
│   ├── depth/                      ← Depth Anything V2 wrapper
│   └── models/                     ← U-Net footprint + MLP models
├── scripts/
│   ├── run_phase29_peak_recovery.py   ← PeakRecoveryMLP training
│   ├── run_phase30_full.py            ← Absolute DSM pipeline
│   ├── run_phase31_mesh_gen.py        ← PyVista 3D mesh generation
│   └── run_phase32_validation.py      ← End-to-end validation
└── runs/                           ← Experiment outputs (not in Git)
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Building MAE (New York zero-shot) | **7.63 ± 0.24 m** |
| Skyscraper height recovery (>40m) | **44.81%** |
| Absolute DSM MAE | **8.14 m** |
| Absolute DSM RMSE | **11.29 m** |
| 3D Mesh vertices per scene | **262,144** |

---

## 🏗️ Pipeline Architecture

```
RGB Satellite Image
        │
        ▼
 Depth Anything V2
 (relative depth)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
 U-Net Building                     Morphological DTM
 Footprint Mask                     Ground Extraction
        │                                      │
        ▼                                      │
 PeakRecoveryMLP                              │
 (Phase 29 — LOCKED)                          │
        │                                      │
        ▼                                      ▼
 Refined nDSM          +            Predicted DTM
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              Absolute DSM (metres)
                       │
                       ▼
           PyVista 3D Textured Mesh
                       │
                       ▼
          Streamlit Interactive Dashboard
```

---

## ⚙️ Supported Input Formats

| Format | Mode | Output |
|--------|------|--------|
| `.tif` / `.tiff` (georeferenced) | **Mode B — Absolute** | Metric DSM in UTM coordinates |
| `.png` / `.jpg` / `.jpeg` | **Mode A — Relative** | Normalised 0–10 m relative DSM |
| `.tif` (no CRS) | Mode A fallback | Relative DSM |

---

## 🔒 Locked Components (Do Not Modify)

- Depth Anything V2 weights
- Phase 29 `PeakRecoveryMLP` checkpoint (`runs/phase29_peak_recovery/`)
- Phase 30 DTM morphological kernel (91 px = 45.5 m)
- Phase 31 PyVista mesh pipeline

---

## 👥 Team

SIH 2024 — Problem Statement: **Satellite Image Height Estimation**
