import os
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.ndimage as ndimage
from rasterio.features import shapes

def apply_morphological_filters(binary_mask):
    print("[INFO] Phase 1 [AI]: Applying morphological smoothing...")
    struct_elem = np.ones((3, 3))
    cleaned = ndimage.binary_opening(binary_mask, structure=struct_elem).astype(binary_mask.dtype)
    cleaned = ndimage.binary_closing(cleaned, structure=struct_elem).astype(cleaned.dtype)
    return cleaned

def extract_and_optimize_polygons(binary_mask, transform, crs, min_area_sqm=1000, simplify_tolerance=5):
    print("[INFO] Phase 1 [GIS]: Extracting and optimizing flood polygons...")
    mask_int = binary_mask.astype(np.uint8)
    
    results = [{'properties': {'class': 'Water'}, 'geometry': geom} 
               for geom, val in shapes(mask_int, mask=mask_int, transform=transform) if val == 1.0]
    
    if not results:
        return gpd.GeoDataFrame()
        
    flood_gdf = gpd.GeoDataFrame.from_features(results, crs=crs)
    flood_gdf_metric = flood_gdf.to_crs("EPSG:3857")
    
    flood_gdf_metric = flood_gdf_metric[flood_gdf_metric.geometry.area > min_area_sqm]
    if not flood_gdf_metric.empty:
        flood_gdf_metric['geometry'] = flood_gdf_metric.geometry.simplify(simplify_tolerance)
        
    return flood_gdf_metric

def calculate_metrics_and_export(flood_gdf_metric, osm_features, output_dir='flood_outputs'):
    print("[INFO] Phase 2: Calculating strategic metrics and exporting data...")
    os.makedirs(output_dir, exist_ok=True)
    
    report_dict = {
        "total_flood_area_sqkm": 0.0,
        "buildings_damaged": 0,
        "roads_damaged_km": 0.0,
        "farmland_damaged_sqkm": 0.0,
        "csv_path": "",
        "geojson_path": ""
    }
    
    if flood_gdf_metric.empty:
        print("[WARNING] No valid flood data to process.")
        return report_dict
        
    report_dict["total_flood_area_sqkm"] = round(flood_gdf_metric.geometry.area.sum() / 1e6, 2)
    
    if not osm_features.empty:
        if 'building' in osm_features.columns:
            bldgs = osm_features[osm_features['building'].notnull()].copy()
            bldgs = bldgs[bldgs.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if not bldgs.empty:
                intersect_bldgs = gpd.overlay(flood_gdf_metric, bldgs, how='intersection')
                report_dict["buildings_damaged"] = len(intersect_bldgs)
                
        if 'highway' in osm_features.columns:
            hwys = osm_features[osm_features['highway'].notnull()].copy()
            hwys = hwys[hwys.geometry.geom_type.isin(['LineString', 'MultiLineString'])]
            if not hwys.empty:
                intersect_hwys = gpd.overlay(flood_gdf_metric, hwys, how='intersection')
                report_dict["roads_damaged_km"] = round(intersect_hwys.geometry.length.sum() / 1000, 2)
                
        if 'landuse' in osm_features.columns:
            farms = osm_features[osm_features['landuse'] == 'farmland'].copy()
            farms = farms[farms.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]
            if not farms.empty:
                intersect_farms = gpd.overlay(flood_gdf_metric, farms, how='intersection')
                report_dict["farmland_damaged_sqkm"] = round(intersect_farms.geometry.area.sum() / 1e6, 2)

    # Save CSV
    df = pd.DataFrame([{
        "Total Flood Area (sq km)": report_dict["total_flood_area_sqkm"],
        "Buildings Flooded": report_dict["buildings_damaged"],
        "Damaged Roads (km)": report_dict["roads_damaged_km"],
        "Affected Farmland (sq km)": report_dict["farmland_damaged_sqkm"]
    }])
    csv_path = os.path.join(output_dir, 'damage_assessment_report.csv')
    df.to_csv(csv_path, index=False)
    report_dict["csv_path"] = csv_path
    
    # Save GeoJSON
    export_gdf = flood_gdf_metric.to_crs("EPSG:4326")
    geojson_path = os.path.join(output_dir, 'flood_boundaries.geojson')
    export_gdf.to_file(geojson_path, driver="GeoJSON")
    report_dict["geojson_path"] = geojson_path
    
    print(f"[SUCCESS] Export completed. Data saved to {output_dir}")
    return report_dict