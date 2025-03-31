import os
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.geometry import Point
from ultralytics import YOLO
from math import atan2, degrees
from tqdm import tqdm

# Constants
IMAGE_DIR = 'data/images'
DEPTH_DIR = 'data/depth'
CSV_PATH = 'data/gps_data.csv'
GEOJSON_PATH = 'data/riseholme_poles_trunk.geojson'
OUTPUT_DIR = 'out/ground'
RADIUS = 10  # radius in meters to search nearby trunks/poles
CLASS_IDS = [2, 4]  # Model class IDs to detect, e.g., 2-Pole, 4-Trunk

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO Model
yolo = YOLO("./models/weights/yolo.pt").to(device)

# Load GPS data and GeoJSON
gps_df = pd.read_csv(CSV_PATH, header=None, delim_whitespace=True,
                     names=['image', 'date', 'time', 'lat', 'lon', 'alt', 'other1', 'other2', 'other3',
                            'other4', 'other5', 'other6', 'other7', 'other8', 'other9'])

features_gdf = gpd.read_file(GEOJSON_PATH)

# Convert trunk/pole locations to geospatial
features_gdf.set_crs(epsg=4326, inplace=True)

# Helper function to calculate heading
def calculate_heading(prev, next):
    dy = next[0] - prev[0]
    dx = next[1] - prev[1]
    angle = atan2(dy, dx)
    heading = (degrees(angle) + 360) % 360
    return heading

# Helper function to find nearby trunks/poles
def find_nearby_features(coord, heading, radius):
    point = Point(coord[1], coord[0])
    buffer = point.buffer(radius / 111139)  # radius in degrees (~111139m = 1 degree)
    nearby = features_gdf[features_gdf.geometry.within(buffer)]
    return nearby

# Helper function to extract patch
def extract_patch(image, center_x, center_y, size=224):
    h, w = image.shape[:2]
    half_size = size // 2

    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, w)
    y2 = min(center_y + half_size, h)

    # Adjust if at edge
    if x2 - x1 < size:
        if x1 == 0:
            x2 = size
        else:
            x1 = w - size
    if y2 - y1 < size:
        if y1 == 0:
            y2 = size
        else:
            y1 = h - size

    patch = image[y1:y2, x1:x2]
    return patch

# Processing Loop
for idx, row in tqdm(gps_df.iterrows(), total=len(gps_df), desc="Processing Images"):
    image_path = os.path.join(IMAGE_DIR, row['image'])
    depth_path = os.path.join(DEPTH_DIR, row['image'])

    # Read image and depth
    if not os.path.exists(image_path) or not os.path.exists(depth_path):
        continue

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Run YOLO detection
    detections = yolo.predict(img, conf=0.2, classes=CLASS_IDS, verbose=False)

    if len(detections) == 0:
        continue

    # Infer heading from previous and next points
    if 0 < idx < len(gps_df) - 1:
        prev_coord = (gps_df.iloc[idx - 1]['lat'], gps_df.iloc[idx - 1]['lon'])
        next_coord = (gps_df.iloc[idx + 1]['lat'], gps_df.iloc[idx + 1]['lon'])
        heading = calculate_heading(prev_coord, next_coord)
    else:
        heading = None  # Skip or handle separately

    nearby_features = find_nearby_features((row['lat'], row['lon']), heading, RADIUS)

    # For each detection
    for det in detections:
        for box in det.boxes.xywh.numpy():
            center_x, center_y = int(box[0]), int(box[1])

            # Extract depth at center
            depth_value = depth[center_y, center_x]

            # Verify if nearby features exist
            if nearby_features.empty:
                continue

            for feature in nearby_features.itertuples():
                feature_id = feature.Index
                feature_dir = os.path.join(OUTPUT_DIR, str(feature_id))
                os.makedirs(feature_dir, exist_ok=True)

                patch = extract_patch(img, center_x, center_y)
                output_filename = os.path.join(feature_dir, f"{row['image']}")
                cv2.imwrite(output_filename, patch)

print("Extraction completed.")

