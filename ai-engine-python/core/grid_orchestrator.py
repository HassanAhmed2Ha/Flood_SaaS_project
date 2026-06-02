"""
Grid-Tiling Orchestrator for Large-Area Flood Detection
========================================================
Splits a large AOI into smaller tiles, fetches satellite data for each tile
concurrently, runs AI inference, then mosaics the results into a single
seamless flood mask.

Optimized for CPU-bound serverless execution.
"""

import os
import math
import uuid
import shutil
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Optional

from core.gee_fetcher import fetch_8_channel_image
from core.ai_segmentation import predict_flood
from core.gis_metrics import apply_morphological_filters


# ---------------------------------------------------------------------------
# 1. Grid Generation
# ---------------------------------------------------------------------------

def generate_grid(
    center_lat: float,
    center_lon: float,
    radius_km: float = 20.0,
    tile_size_km: float = 5.0,
) -> List[dict]:
    """
    Generate a grid of tile bounding boxes covering a circular AOI.
    
    Each tile is returned as a dict with:
        lat, lon  — tile center coordinates
        idx       — (row, col) tuple for logging
    
    The grid covers a square that circumscribes the target circle.
    """
    # Approximate degree offsets (WGS-84 mid-latitude)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))

    half_extent_lat = radius_km / km_per_deg_lat
    half_extent_lon = radius_km / km_per_deg_lon

    tile_step_lat = tile_size_km / km_per_deg_lat
    tile_step_lon = tile_size_km / km_per_deg_lon

    # Number of tiles in each direction from center
    n_lat = math.ceil(radius_km / tile_size_km)
    n_lon = math.ceil(radius_km / tile_size_km)

    tiles = []
    for row in range(-n_lat, n_lat + 1):
        for col in range(-n_lon, n_lon + 1):
            tile_lat = center_lat + row * tile_step_lat
            tile_lon = center_lon + col * tile_step_lon

            # Skip tiles whose center falls outside the circle
            dlat = (tile_lat - center_lat) * km_per_deg_lat
            dlon = (tile_lon - center_lon) * km_per_deg_lon
            if math.sqrt(dlat**2 + dlon**2) > radius_km + tile_size_km * 0.7:
                continue

            tiles.append({
                "lat": round(tile_lat, 6),
                "lon": round(tile_lon, 6),
                "idx": (row + n_lat, col + n_lon),
            })

    print(f"[GRID] Generated {len(tiles)} tiles "
          f"({2*n_lat+1}×{2*n_lon+1} grid, {tile_size_km}km each, "
          f"{radius_km}km radius)")
    return tiles


# ---------------------------------------------------------------------------
# 2. Tile Worker (GEE fetch + AI inference for ONE tile)
# ---------------------------------------------------------------------------

def _process_single_tile(
    tile: dict,
    model,
    start_date: str,
    end_date: str,
    workspace: str,
) -> dict:
    """
    Worker function executed per-tile:
      1. Fetch 8-channel Sentinel data via GEE
      2. Run predict_flood on the downloaded TIF
      3. Write the predicted mask as a single-band GeoTIFF
      4. Return the result dict (or empty mask on failure)
    """
    row, col = tile["idx"]
    tag = f"[TILE {row},{col}]"
    tile_id = f"tile_{row}_{col}_{uuid.uuid4().hex[:6]}"
    tif_name = f"{tile_id}.tif"
    mask_path = os.path.join(workspace, f"{tile_id}_mask.tif")

    try:
        # --- Step A: Fetch satellite data ---
        print(f"{tag} Fetching satellite data for ({tile['lat']}, {tile['lon']})...")
        tif_path = fetch_8_channel_image(
            tile["lat"], tile["lon"],
            start_date, end_date,
            tif_name,
            output_dir=workspace,
        )
        if not tif_path:
            print(f"{tag} SKIP — no satellite data available.")
            return {"status": "empty", "mask_path": None, "confidence": 0.0}

        # --- Step B: AI Inference ---
        print(f"{tag} Running AI inference...")
        raw_mask, transform, crs, confidence, _ = predict_flood(model, tif_path)

        # --- Step C: Morphological cleaning ---
        cleaned_mask = apply_morphological_filters(raw_mask)

        # --- Step D: Write mask as georeferenced GeoTIFF ---
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()

        profile.update(
            count=1,
            dtype='float32',
            nodata=0.0,
        )
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(cleaned_mask.astype(np.float32), 1)

        print(f"{tag} OK — confidence={confidence:.1f}%, "
              f"flood_pixels={int(np.sum(cleaned_mask))}")
        return {
            "status": "ok",
            "mask_path": mask_path,
            "confidence": confidence,
            "flood_pixels": int(np.sum(cleaned_mask)),
        }

    except Exception as e:
        print(f"{tag} ERROR — {e}")
        return {"status": "error", "mask_path": None, "confidence": 0.0}


# ---------------------------------------------------------------------------
# 3. Concurrent Executor + Mosaicking
# ---------------------------------------------------------------------------

def run_grid_analysis(
    model,
    center_lat: float,
    center_lon: float,
    start_date: str,
    end_date: str,
    radius_km: float = 20.0,
    tile_size_km: float = 5.0,
    max_workers: int = 4,
) -> Tuple[Optional[str], float, dict]:
    """
    Main orchestrator:
      1. Generate grid tiles
      2. Process tiles concurrently (ThreadPool for I/O-bound GEE calls)
      3. Mosaic all per-tile masks into one seamless GeoTIFF
      4. Return (mosaic_path, overall_confidence, summary_dict)
    """
    # Workspace for this run
    run_id = uuid.uuid4().hex[:8]
    workspace = os.path.join("live_data", f"grid_{run_id}")
    os.makedirs(workspace, exist_ok=True)

    # --- Step 1: Generate grid ---
    tiles = generate_grid(center_lat, center_lon, radius_km, tile_size_km)
    if not tiles:
        return None, 0.0, {"total_tiles": 0, "ok": 0, "failed": 0}

    # --- Step 2: Process tiles concurrently ---
    # ThreadPoolExecutor is used because the bottleneck is I/O (GEE download).
    # The TensorFlow model.predict() calls release the GIL, so threads work.
    results = []
    print(f"[GRID] Starting concurrent processing with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_tile, tile, model, start_date, end_date, workspace
            ): tile
            for tile in tiles
        }
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5-min timeout per tile
            except Exception as e:
                tile = futures[future]
                print(f"[GRID] Tile {tile['idx']} timed out: {e}")
                result = {"status": "error", "mask_path": None, "confidence": 0.0}
            results.append(result)

    # --- Step 3: Mosaic successful masks ---
    ok_results = [r for r in results if r["status"] == "ok" and r["mask_path"]]
    failed_count = len(results) - len(ok_results)

    summary = {
        "total_tiles": len(tiles),
        "ok": len(ok_results),
        "failed": failed_count,
    }

    if not ok_results:
        print("[GRID] No tiles produced valid masks. Aborting mosaic.")
        return None, 0.0, summary

    print(f"[GRID] Mosaicking {len(ok_results)} tile masks...")

    # Open all mask rasters
    src_files = [rasterio.open(r["mask_path"]) for r in ok_results]
    mosaic_arr, mosaic_transform = merge(src_files)

    # Close source files
    for s in src_files:
        s.close()

    # Write the final mosaic
    mosaic_path = os.path.join("live_data", f"mosaic_{run_id}.tif")
    mosaic_profile = {
        "driver": "GTiff",
        "height": mosaic_arr.shape[1],
        "width": mosaic_arr.shape[2],
        "count": 1,
        "dtype": "float32",
        "crs": src_files[0].crs if src_files else "EPSG:32610",
        "transform": mosaic_transform,
        "nodata": 0.0,
    }
    # Re-read CRS from first successful tile (src_files already closed)
    with rasterio.open(ok_results[0]["mask_path"]) as ref:
        mosaic_profile["crs"] = ref.crs

    with rasterio.open(mosaic_path, "w", **mosaic_profile) as dst:
        dst.write(mosaic_arr[0], 1)

    # --- Step 4: Calculate aggregate statistics ---
    total_flood_pixels = sum(r.get("flood_pixels", 0) for r in ok_results)
    confidences = [r["confidence"] for r in ok_results if r["confidence"] > 0]
    overall_confidence = float(np.mean(confidences)) if confidences else 0.0

    summary["flood_pixels"] = total_flood_pixels
    summary["mosaic_path"] = mosaic_path

    print(f"[GRID] Mosaic complete: {mosaic_path}")
    print(f"[GRID] Tiles OK={len(ok_results)}, Failed={failed_count}, "
          f"Confidence={overall_confidence:.2f}%")

    # Cleanup per-tile workspace
    try:
        shutil.rmtree(workspace)
    except Exception:
        pass

    return mosaic_path, overall_confidence, summary
