import os
import ee
import geemap
from google.oauth2 import service_account

try:
    key_path = 'auth/gee_service_account.json'
    if not os.path.exists(key_path):
        key_path = os.path.join('ai-engine-python', 'auth', 'gee_service_account.json')
    
    credentials = service_account.Credentials.from_service_account_file(key_path)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
    ee.Initialize(credentials=scoped_credentials, project='flood-saas-project')
    print("[INFO] Google Earth Engine Initialized successfully.")
except Exception as e:
    print(f"[ERROR] GEE Initialization failed: {e}")

def fetch_8_channel_image(lat, lon, start_date, end_date, output_filename, output_dir='live_data'):
    
    
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(4000).bounds()
    
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(roi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.eq('instrumentMode', 'IW'))
                     .select(['VV', 'VH']))
    
    if s1_collection.size().getInfo() == 0:
        print("[ERROR] No Sentinel-1 data found.")
        return None
        
    s1_img = s1_collection.mosaic().clip(roi)
    
    s2_collection = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                     .filterBounds(roi)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                     .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']))
    
    if s2_collection.size().getInfo() == 0:
        print("[WARNING] No clear Sentinel-2 data found. Using empty optical bands.")
        s2_img = ee.Image.constant([0,0,0,0,0,0]).rename(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).clip(roi)
    else:
        s2_img = s2_collection.sort('CLOUDY_PIXEL_PERCENTAGE').first().clip(roi)
    
    combined_8_channel = s1_img.addBands(s2_img).select(['VV', 'VH', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12']).toFloat()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Exporting 8-channel TIF to {out_path}...")
    try:
        geemap.ee_export_image(combined_8_channel, filename=out_path, scale=30, region=roi)
    except Exception as e:
        print(f"[ERROR] Google Earth Engine Export Failed: {e}")
        return None
        
    if not os.path.exists(out_path):
        print("[FATAL ERROR] The image was not downloaded. Pipeline aborted.")
        return None
        
    print("[SUCCESS] Download complete.")
    return out_path