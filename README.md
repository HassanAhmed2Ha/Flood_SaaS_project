<div align="center">

# 🛰️ Flood Intelligence AI

### Satellite-Driven Flood Detection & Tactical Damage Assessment

**U-Net Deep Learning** · **Sentinel-1/2 SAR+Optical Fusion** · **Wide-Area Grid Tiling**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=node.js&logoColor=white)](https://nodejs.org)
[![Three.js](https://img.shields.io/badge/Three.js-r161-000?logo=three.js)](https://threejs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

- [Philosophy & Vision](#philosophy--vision)
- [System Architecture](#system-architecture)
- [Core Methodology & AI Logic](#core-methodology--ai-logic)
- [Directory Structure & File Roles](#directory-structure--file-roles)
- [Setup & Execution](#setup--execution)
- [API Reference](#api-reference)
- [License](#license)

---

## Philosophy & Vision

**Flood Intelligence AI** is an end-to-end geospatial intelligence platform that detects flood extent from satellite imagery, quantifies infrastructure damage, and renders the results inside a cinematic **Tactical Mission Control** interface.

### Design Aesthetic

The frontend rejects conventional dashboard design in favor of a **Cyberpunk / Ghost-in-the-Shell** tactical HUD — a cardless, data-layer architecture where information floats over a 3D Earth globe rendered with Three.js. Corner brackets replace card borders. Monospaced typography (`JetBrains Mono`, `Space Mono`) replaces sans-serif defaults. Cyan neon-glow accents and glassmorphism panels replace flat Material boxes.

When a scan completes, a choreographed camera interpolation zooms the globe into the target coordinates, cross-fading into a 2D Leaflet tactical map where georeferenced flood polygons are overlaid in real time.

### AI Philosophy

The inference engine combines the **pattern recognition** of a trained U-Net deep learning model with the **physical grounding** of Synthetic Aperture Radar (SAR) backscatter thresholds. The model ingests an 8-channel composite — 2 SAR polarizations (VV, VH) fused with 6 optical spectral bands (B2, B3, B4, B8, B11, B12) — enabling it to distinguish between permanent water bodies, agricultural flooding, and urban inundation.

A **Smart Cloud Fallback** safety net prevents model hallucination when dense cloud cover eliminates all optical signal, while preserving legitimate double-bounce flood signals in vegetated agricultural regions.

### Architecture Philosophy

Wide-area disaster scanning (up to **400 km²** per request) is achieved through a **CPU-optimized Grid-Tiling Engine** that subdivides the area of interest into 5×5 km tiles, fetches satellite data concurrently via `ThreadPoolExecutor`, runs per-tile AI inference, and mosaics the results into a seamless output raster. This design eliminates Earth Engine timeout failures and operates without GPU dependencies — suitable for serverless or containerized deployment.

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Port 3000)                         │
│  Vanilla JS · Three.js Globe · Leaflet Map · Tactical HUD       │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────────────┐   │
│  │ Left Panel  │ │ 3D Globe /   │ │ Right Panel              │   │
│  │ Mission     │ │ 2D Tactical  │ │ AI Confidence Gauge      │   │
│  │ Parameters  │ │ Map View     │ │ Damage Metrics           │   │
│  └─────────────┘ └──────────────┘ └──────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Bottom Panel — System Log (auto-scrolling telemetry stream) │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────┬────────────────────────────────────────────┘
                       │  POST /api/scan
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│               NODE.JS API GATEWAY (Port 3000)                    │
│  Express.js — CORS, JSON parsing, static file serving            │
│  Forwards request body to Python AI Engine via axios             │
└──────────────────────┬───────────────────────────────────────────┘
                       │  POST /api/v1/analyze_flood
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│              PYTHON FASTAPI AI ENGINE (Port 8000)                │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │ GEE      │──▶│ AI           │──▶│ GIS Post-Processing      │  │
│  │ Fetcher  │   │ Segmentation │   │ Morphological Filtering  │  │
│  │          │   │ (U-Net)      │   │ Polygon Extraction       │  │
│  └──────────┘   └──────────────┘   │ OSM Damage Assessment    │  │
│                                    └──────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │         Grid Orchestrator (Wide-Area Tiling Mode)            │ │
│  │  generate_grid → ThreadPoolExecutor → rasterio.merge         │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  Google Earth Engine    │
          │  Sentinel-1 SAR (GRD)  │
          │  Sentinel-2 MSI        │
          └─────────────────────────┘
```

### Request Lifecycle

1. **User Input** → The operator enters target coordinates, date range, and scan radius on the Left Panel.
2. **Frontend** → `index.js` fires a `POST /api/scan` with the payload (`latitude`, `longitude`, `start_date`, `end_date`, `radius_km`, `tile_size_km`, `max_workers`). A simulated tactical progress stream animates the System Log during the wait.
3. **Node.js Gateway** → `server.js` receives the JSON, logs it, and proxies it verbatim to the Python engine at `http://127.0.0.1:8000/api/v1/analyze_flood`.
4. **Python Engine** → `main.py` branches:
   - **Single-tile mode** (`radius_km = 0`): Calls `gee_fetcher` → `predict_flood` → `gis_metrics` sequentially.
   - **Grid mode** (`radius_km > 0`): Delegates to `grid_orchestrator.run_grid_analysis()`, which tiles the AOI, processes concurrently, and mosaics.
5. **Response** → The engine returns JSON with `confidence_score`, `metrics` (flood area, buildings damaged, roads affected), and `flood_geojson`.
6. **Frontend Render** → On success, a cinematic 3D-to-2D transition zooms into the target, Leaflet renders the flood GeoJSON, and the Right Panel displays tactical intel gauges.

---

## Core Methodology & AI Logic

### 1. Multi-Sensor Data Ingestion

| Channel | Source | Band | Purpose |
|---------|--------|------|---------|
| 0 | Sentinel-1 SAR | **VV** | Vertical-transmit/Vertical-receive backscatter |
| 1 | Sentinel-1 SAR | **VH** | Vertical-transmit/Horizontal-receive backscatter |
| 2 | Sentinel-2 MSI | **B2** | Blue (490 nm) |
| 3 | Sentinel-2 MSI | **B3** | Green (560 nm) |
| 4 | Sentinel-2 MSI | **B4** | Red (665 nm) |
| 5 | Sentinel-2 MSI | **B8** | NIR (842 nm) |
| 6 | Sentinel-2 MSI | **B11** | SWIR-1 (1610 nm) |
| 7 | Sentinel-2 MSI | **B12** | SWIR-2 (2190 nm) |

SAR channels are normalized from `[-25, 0] dB` → `[0, 1]`. Optical channels are clipped at `3000` and divided to `[0, 1]`.

### 2. Patchify / Unpatchify Pipeline

The 8-channel GeoTIFF is padded to the nearest multiple of 256, then decomposed into **256×256×8 patches** using `patchify`. Each patch is fed to the U-Net independently. The per-patch probability maps are reassembled via `unpatchify`, then cropped back to the original image dimensions.

### 3. Smart Cloud Fallback

```
IF max(optical_bands[2:8]) == 0.0:
    ⚠ Dense cloud cover detected — optical data is missing
    → Apply strict SAR threshold: mask out pixels where VV > -14.0 dB
    → Prevents U-Net from hallucinating 100% flood masks

ELSE:
    ✓ Optical data present — trust the U-Net prediction as-is
    → Preserves agricultural double-bounce flood signals
```

This conditional safety net was engineered specifically to handle the failure mode where the U-Net, trained on 8-channel data, saturates to full-flood predictions when 6 of its 8 input channels are zeroed out by cloud cover.

### Visual Validation: Hybrid SAR Fallback

![Radar Validation Histogram](./image.png)

*Fig 1: The engine successfully applying the SAR physical threshold (-15 dB) during a 100% cloud-cover event (Optical RGB is blank), accurately isolating flood polygons from background land.*

### 4. Post-Processing Pipeline

1. **Morphological Smoothing** — Binary opening + closing with a 3×3 structuring element removes salt-and-pepper noise.
2. **Polygon Extraction** — `rasterio.features.shapes()` converts the binary mask to vector polygons.
3. **Area Filtering** — Polygons smaller than 1000 m² are discarded.
4. **Geometry Simplification** — Douglas-Peucker simplification (5m tolerance) reduces polygon vertex count for efficient frontend rendering.
5. **Infrastructure Assessment** — OpenStreetMap data (via OSMnx) is spatially intersected with flood polygons to quantify affected buildings, road segments, and agricultural land.

### 5. Grid-Tiling Engine

For scans with `radius_km > 0`:

1. **Grid Generation** — `generate_grid()` calculates tile centers in a WGS-84 grid pattern, filtering tiles whose centers fall outside the circular AOI.
2. **Concurrent Processing** — `ThreadPoolExecutor` dispatches tile workers that independently fetch GEE data and run inference. Thread-based parallelism is used because the bottleneck is I/O (GEE downloads), and TensorFlow's `model.predict()` releases the GIL.
3. **Raster Mosaicking** — `rasterio.merge.merge()` combines all per-tile mask GeoTIFFs into a single seamless flood mask.
4. **Cleanup** — Per-tile workspace directories are removed after mosaicking.

---

## Directory Structure & File Roles

```
Flood_SaaS_Project/
│
├── frontend/src/                    # Client — Tactical Mission Control UI
│   ├── index.html                   # Main HTML shell (3-panel CSS Grid layout)
│   ├── index.js                     # Core logic: Three.js globe, Leaflet map,
│   │                                #   scan handler, transition choreography,
│   │                                #   tactical progress stream, dossier loader
│   ├── tactical.css                 # Full design system: CSS Grid layout, HUD,
│   │                                #   glassmorphism panels, neon-glow accents,
│   │                                #   dossier modal, CRT scan-line animations
│   ├── about_component.html         # Dynamically-loaded Intel Dossier modal
│   ├── getEarthMat.js               # Three.js Earth material (day/night textures)
│   ├── getFresnelMat.js             # Atmospheric Fresnel glow shader
│   ├── getLayer.js                  # Cloud/city-light overlay layers
│   ├── getStarfield.js              # Procedural star particle system
│   └── textures/                    # Earth, cloud, and night-light texture maps
│
├── backend-node/                    # API Gateway — Express.js
│   ├── server.js                    # Static file server + /api/scan proxy to Python
│   └── package.json                 # Dependencies: express, axios, cors
│
├── ai-engine-python/                # AI Engine — FastAPI + TensorFlow
│   ├── main.py                      # FastAPI app: /api/v1/analyze_flood endpoint,
│   │                                #   dual-mode branching (single-tile vs grid),
│   │                                #   model lifecycle management
│   ├── Dockerfile                   # Container configuration
│   ├── core/
│   │   ├── gee_fetcher.py           # Google Earth Engine interface: fetches 8-channel
│   │   │                            #   Sentinel-1 + Sentinel-2 composites as GeoTIFF
│   │   ├── ai_segmentation.py       # U-Net inference: patchify, predict, unpatchify,
│   │   │                            #   Smart Cloud Fallback, confidence scoring
│   │   ├── gis_metrics.py           # Morphological filtering, polygon extraction,
│   │   │                            #   OSM damage assessment, GeoJSON export
│   │   └── grid_orchestrator.py     # Wide-area grid tiling: tile generation,
│   │                                #   concurrent execution, raster mosaicking
│   ├── models/                      # Pre-trained U-Net weights (.keras)
│   ├── auth/                        # GEE service account credentials
│   ├── live_data/                   # Runtime satellite downloads & inference outputs
│   ├── cache/                       # Cached intermediate results
│   └── flood_outputs/               # Exported flood masks and GeoJSON
│
└── requirements.txt                 # Python dependency manifest
```

---

## Setup & Execution

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ |
| **Node.js** | 18+ |
| **npm** | 9+ |
| **Google Earth Engine** | Authenticated service account |

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Flood_SaaS_Project.git
cd Flood_SaaS_Project
```

### 2. Configure Google Earth Engine

Place your GEE service account credentials at:

```
ai-engine-python/auth/gee_service_account.json
```

Update the project ID in `core/gee_fetcher.py`:

```python
ee.Initialize(project='your-gee-project-id')
```

### 3. Install & Start the Python AI Engine

```bash
cd ai-engine-python
pip install -r ../requirements.txt
python main.py
```

The engine will:
- Load the U-Net model into memory
- Start the FastAPI server on `http://localhost:8000`
- Display `[API] Model loaded successfully. Server is READY.`

### 4. Install & Start the Node.js Gateway

```bash
cd backend-node
npm install
node server.js
```

The gateway will:
- Serve the frontend static files
- Proxy `/api/scan` requests to the Python engine
- Start on `http://localhost:3000`

### 5. Open the Application

Navigate to **`http://localhost:3000`** in your browser. You will see the 3D Earth globe with the Mission Parameters panel on the left.

---

## API Reference

### `POST /api/v1/analyze_flood`

#### Request Body

```json
{
  "latitude": 13.5,
  "longitude": 33.3,
  "start_date": "2021-11-16",
  "end_date": "2021-11-26",
  "radius_km": 20,
  "tile_size_km": 5,
  "max_workers": 4
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `latitude` | `float` | *required* | Target latitude (WGS-84) |
| `longitude` | `float` | *required* | Target longitude (WGS-84) |
| `start_date` | `string` | *required* | Observation window start (YYYY-MM-DD) |
| `end_date` | `string` | *required* | Observation window end (YYYY-MM-DD) |
| `radius_km` | `float` | `0` | Scan radius. `0` = single-tile legacy mode |
| `tile_size_km` | `float` | `5.0` | Grid tile dimension (km) |
| `max_workers` | `int` | `4` | Concurrent thread count for grid mode |

#### Response

```json
{
  "confidence_score": 99.73,
  "metrics": {
    "total_flood_area_sqkm": 42.7,
    "buildings_damaged": 1247,
    "roads_affected_km": 18.3,
    "farmland_affected_sqkm": 31.2
  },
  "flood_geojson": { "type": "FeatureCollection", "features": [...] },
  "grid_summary": { "total_tiles": 49, "ok": 47, "failed": 2 }
}
```

---

## License

This project is released under the **MIT License**.

---

<div align="center">
  <sub>Built by <strong>Hassan</strong> — AI & Remote Sensing Engineer</sub>
</div>
