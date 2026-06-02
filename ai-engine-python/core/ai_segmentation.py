import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras import backend as K
from patchify import patchify, unpatchify

def calculate_iou(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred > 0.4, tf.float32) 
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def focal_dice_loss(y_true, y_pred):
    return K.mean(y_pred)

def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(2):
        clipped = np.clip(img[:,:,i], -25.0, 0.0)
        img[:,:,i] = (clipped - (-25.0)) / (0.0 - (-25.0))
    for i in range(2, 8):
        clipped = np.clip(img[:,:,i], 0.0, 3000.0)
        img[:,:,i] = clipped / 3000.0
    return img

def load_flood_model(model_path):
    print("[INFO] Loading U-Net Model into memory...")
    custom_objs = {'focal_dice_loss': focal_dice_loss, 'calculate_iou': calculate_iou}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objs)

def predict_flood(model, image_path, patch_size=256):
    print("[INFO] Running AI segmentation with Land Cover & Confidence Scoring...")
    with rasterio.open(image_path) as src:
        large_image = src.read()
        transform = src.transform
        crs = src.crs
        
    large_image = np.moveaxis(large_image, 0, -1)
    original_shape = large_image.shape
    
    pad_h = (patch_size - (original_shape[0] % patch_size)) % patch_size
    pad_w = (patch_size - (original_shape[1] % patch_size)) % patch_size
    padded_image = np.pad(large_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    padded_shape = padded_image.shape
    
    padded_image = normalize_image(padded_image)
    patches = patchify(padded_image, (patch_size, patch_size, 8), step=patch_size)
    
    predicted_patches_binary = np.zeros((patches.shape[0], patches.shape[1], 1, patch_size, patch_size, 1))
    predicted_patches_prob = np.zeros((patches.shape[0], patches.shape[1], 1, patch_size, patch_size, 1))
    
    PREDICTION_THRESHOLD = 0.50 
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, 0, :, :, :]
            input_tensor = np.expand_dims(single_patch, axis=0)
            
            pred_prob = model.predict(input_tensor, verbose=0)[0]
            predicted_patches_prob[i, j, 0, :, :, :] = pred_prob
            predicted_patches_binary[i, j, 0, :, :, :] = (pred_prob > PREDICTION_THRESHOLD).astype(np.float32)

    reconstructed_binary = unpatchify(predicted_patches_binary, (padded_shape[0], padded_shape[1], 1))
    reconstructed_prob = unpatchify(predicted_patches_prob, (padded_shape[0], padded_shape[1], 1))
    
    final_mask = reconstructed_binary[:original_shape[0], :original_shape[1], 0]
    confidence_map = reconstructed_prob[:original_shape[0], :original_shape[1], 0]
    
    raw_vv = large_image[:, :, 0]
    
    # 1. Edge/No-Data Validation
    final_mask[raw_vv == 0.0] = 0
    
    # 2. Smart Cloud Fallback — only activates when optical bands are missing
    optical_bands = large_image[:, :, 2:8]  # Channels 2-7: B2, B3, B4, B8, B11, B12
    optical_is_empty = np.max(optical_bands) == 0.0
    
    if optical_is_empty:
        print("[WARNING] Optical data missing. Applying strict SAR threshold to prevent model saturation.")
        final_mask[(raw_vv > -14.0) & (raw_vv != 0.0)] = 0
    else:
        # Optical data present — trust the U-Net prediction as-is
        # (preserves double-bounce vegetation/agricultural flood signals)
        pass
    
    valid_flood_pixels = confidence_map[final_mask == 1.0]
    overall_confidence = 0.0
    
    if len(valid_flood_pixels) > 0:
        overall_confidence = np.mean(valid_flood_pixels) * 100
        print(f"[SUCCESS] AI Flood Confidence Score: {overall_confidence:.2f}%")
    else:
        print("[INFO] No flood detected after physical validation.")

    return final_mask, transform, crs, overall_confidence, confidence_map