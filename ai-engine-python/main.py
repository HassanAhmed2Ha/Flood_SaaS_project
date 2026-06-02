from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn
import os
import rasterio
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box
import json
import re
from datetime import datetime
import uuid

# Core modules
from core.gee_fetcher import fetch_8_channel_image
from core.ai_segmentation import load_flood_model, predict_flood
from core.gis_metrics import apply_morphological_filters, extract_and_optimize_polygons, calculate_metrics_and_export
from core.grid_orchestrator import run_grid_analysis

# 1. Initialize FastAPI
app = FastAPI(title="Flood Intelligence AI Engine", version="2.0")

# Global model reference (loaded once at startup)
flood_model = None

# Global tasks store for background processing state
tasks = {}

# 2. Load model into memory on startup
@app.on_event("startup")
async def startup_event():
    global flood_model
    print("[API] Starting up: Loading AI Model into Memory...")
    MODEL_PATH = 'models/unet_flood_model.keras'
    try:
        flood_model = load_flood_model(MODEL_PATH)
        print("[API] Model loaded successfully. Server is READY.")
    except Exception as e:
        print(f"[API] CRITICAL ERROR loading model: {e}")


# 3. Request schema
class ScanRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    start_date: str
    end_date: str
    # Grid-tiling parameters (optional — defaults to single-tile mode)
    radius_km: float = Field(0.0, ge=0.0, le=50.0)      # Max 50km to prevent resource exhaustion
    tile_size_km: float = Field(5.0, ge=1.0, le=20.0)    # Min 1km to prevent excessive tiles
    max_workers: int = Field(4, ge=1, le=16)

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: str) -> str:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError("Date must be in YYYY-MM-DD format.")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date: {v}")
        return v

    @model_validator(mode='after')
    def validate_date_range(self) -> 'ScanRequest':
        try:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
            if end < start:
                raise ValueError("end_date must be after or equal to start_date.")
        except Exception as e:
            raise ValueError(str(e))
        return self


# 4. Background task runner
def run_analyze_flood_in_background(task_id: str, request: ScanRequest):
    global flood_model
    try:
        if flood_model is None:
            raise Exception("AI Model is not loaded. Check server logs.")

        # ====================================================================
        #  BRANCH A: Grid-Tiling Mode (radius_km > 0)
        # ====================================================================
        if request.radius_km > 0:
            print(f"[Background Task] GRID MODE — radius={request.radius_km}km, "
                  f"tile={request.tile_size_km}km, workers={request.max_workers}")

            mosaic_path, overall_confidence, summary = run_grid_analysis(
                model=flood_model,
                center_lat=request.latitude,
                center_lon=request.longitude,
                start_date=request.start_date,
                end_date=request.end_date,
                radius_km=request.radius_km,
                tile_size_km=request.tile_size_km,
                max_workers=request.max_workers,
            )

            if not mosaic_path:
                raise Exception(f"Grid analysis failed. {summary.get('failed', 0)} tiles returned no data.")

            # Extract polygons from the mosaic mask
            try:
                with rasterio.open(mosaic_path) as src:
                    mosaic_mask = src.read(1)
                    mosaic_transform = src.transform
                    mosaic_crs = src.crs
                    bounds = src.bounds

                flood_gdf = extract_and_optimize_polygons(mosaic_mask, mosaic_transform, mosaic_crs)

                # OSM infrastructure analysis on the mosaic extent
                img_poly = box(*bounds)
                bounds_gdf = gpd.GeoDataFrame({'geometry': [img_poly]}, crs=mosaic_crs)
                img_poly_4326 = bounds_gdf.to_crs("EPSG:4326").geometry.iloc[0]

                try:
                    osm_features = ox.features_from_polygon(
                        img_poly_4326,
                        tags={'building': True, 'highway': True, 'landuse': ['farmland']}
                    ).to_crs("EPSG:3857").reset_index(drop=True)
                except Exception as e:
                    print(f"[WARNING] OSM fetch on mosaic extent: {e}")
                    osm_features = gpd.GeoDataFrame()

                report_dict = calculate_metrics_and_export(flood_gdf, osm_features, output_dir='flood_outputs')

            except Exception as e:
                raise Exception(f"Mosaic post-processing error: {str(e)}")

            geojson_data = None
            if os.path.exists(report_dict.get("geojson_path", "")):
                with open(report_dict["geojson_path"], "r") as f:
                    geojson_data = json.load(f)

            tasks[task_id] = {
                "status": "completed",
                "data": {
                    "status": "success",
                    "mode": "grid",
                    "confidence_score": round(overall_confidence, 2),
                    "grid_summary": {
                        "total_tiles": summary["total_tiles"],
                        "tiles_ok": summary["ok"],
                        "tiles_failed": summary["failed"],
                    },
                    "metrics": {
                        "total_flood_area_sqkm": report_dict.get("total_flood_area_sqkm", 0),
                        "buildings_damaged": report_dict.get("buildings_damaged", 0),
                        "roads_damaged_km": report_dict.get("roads_damaged_km", 0),
                        "farmland_damaged_sqkm": report_dict.get("farmland_damaged_sqkm", 0),
                    },
                    "geojson": geojson_data,
                }
            }

        # ====================================================================
        #  BRANCH B: Single-Tile Mode (legacy — radius_km == 0)
        # ====================================================================
        else:
            print("[Background Task] SINGLE-TILE MODE (legacy)")
            output_tif = f"flood_{request.latitude}_{request.longitude}.tif"

            # الخطوة 1: جلب بيانات القمر الصناعي
            tif_path = fetch_8_channel_image(
                request.latitude,
                request.longitude,
                request.start_date,
                request.end_date,
                output_tif
            )

            if not tif_path:
                raise Exception("Failed to fetch satellite data. Clouds too dense or wrong dates.")

            # الخطوة 2: تشغيل الذكاء الاصطناعي والفلاتر الفيزيائية
            try:
                raw_ai_mask, img_transform, img_crs, ai_confidence, conf_map = predict_flood(flood_model, tif_path)
            except Exception as e:
                raise Exception(f"AI Processing Error: {str(e)}")

            # الخطوة 3: المعالجة الجغرافية (GIS)
            cleaned_mask = apply_morphological_filters(raw_ai_mask)
            flood_metric_gdf = extract_and_optimize_polygons(cleaned_mask, img_transform, img_crs)

            # الخطوة 4: جلب بيانات البنية التحتية من OpenStreetMap
            print("[INFO] Fetching infrastructure data from OSM...")
            try:
                with rasterio.open(tif_path) as src:
                    bounds = src.bounds
                img_poly = box(*bounds)
                bounds_gdf = gpd.GeoDataFrame({'geometry': [img_poly]}, crs=img_crs)
                img_poly_4326 = bounds_gdf.to_crs("EPSG:4326").geometry.iloc[0]

                osm_features = ox.features_from_polygon(
                    img_poly_4326,
                    tags={'building': True, 'highway': True, 'landuse': ['farmland']}
                ).to_crs("EPSG:3857").reset_index(drop=True)
            except Exception as e:
                print(f"[WARNING] No OSM infrastructure found or error: {e}")
                osm_features = gpd.GeoDataFrame()

            # الخطوة 5: حساب الخسائر وتصدير الملفات
            report_dict = calculate_metrics_and_export(flood_metric_gdf, osm_features, output_dir='flood_outputs')

            # الخطوة 6: قراءة ملف GeoJSON ودمجه في الرد المباشر
            geojson_data = None
            if os.path.exists(report_dict.get("geojson_path", "")):
                with open(report_dict["geojson_path"], "r") as f:
                    geojson_data = json.load(f)

            tasks[task_id] = {
                "status": "completed",
                "data": {
                    "status": "success",
                    "mode": "single",
                    "confidence_score": round(ai_confidence, 2),
                    "metrics": {
                        "total_flood_area_sqkm": report_dict.get("total_flood_area_sqkm", 0),
                        "buildings_damaged": report_dict.get("buildings_damaged", 0),
                        "roads_damaged_km": report_dict.get("roads_damaged_km", 0),
                        "farmland_damaged_sqkm": report_dict.get("farmland_damaged_sqkm", 0)
                    },
                    "geojson": geojson_data
                }
            }

    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}


# 5. Main analysis endpoint (scheduling background task)
@app.post("/api/v1/analyze_flood")
async def analyze_flood(request: ScanRequest, background_tasks: BackgroundTasks):
    global flood_model

    if flood_model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded. Check server logs.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(run_analyze_flood_in_background, task_id, request)

    return {"task_id": task_id}


# 6. Task status endpoint
@app.get("/api/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]



# Run server
if __name__ == "__main__":
    print("Starting FastAPI Server on Port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)