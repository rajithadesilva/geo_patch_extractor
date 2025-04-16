import os
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from shapely.geometry import Point
from ultralytics import YOLO
from math import atan2, degrees, radians, cos, sin
from tqdm import tqdm

# Constants
IMAGE_DIR = 'data/colour'
DEPTH_DIR = 'data/depth'
CSV_PATH = 'data/1_6.csv'
GEOJSON_PATH = 'data/riseholme_poles_trunk.geojson'
OUTPUT_DIR = 'out'
YOLO_OUTPUT_DIR = 'out/yolo'
RADIUS = 5  # radius in meters to search nearby trunks/poles
CLASS_IDS = [2, 4]  # Model class IDs to detect, e.g., 2-Pole, 4-Trunk
CLASS_NAMES = {2: 'poles', 4: 'trunks'}

# Determine device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO Model
yolo = YOLO("./models/yolo.pt").to(device)

# Load GPS data and GeoJSON
gps_df = pd.read_csv(CSV_PATH, delim_whitespace=False)
features_gdf = gpd.read_file(GEOJSON_PATH)
features_gdf.set_crs(epsg=4326, inplace=True)

# Helper function to calculate heading
def calculate_heading(prev, next):
    dy = next[0] - prev[0]
    dx = next[1] - prev[1]
    angle = atan2(dy, dx)
    heading = (degrees(angle) + 360) % 360
    return heading

# Helper function to find forward-facing nearby features
def find_forward_facing_features(coord, heading_deg, radius):
    lon, lat = coord[1], coord[0]
    angle_rad = radians(heading_deg)
    fov_angle = radians(90)  # 90-degree field of view (forward half)
    num_points = 100
    sector_points = []
    for i in range(num_points + 1):
        angle = angle_rad - fov_angle / 2 + i * fov_angle / num_points
        dx = radius * cos(angle)
        dy = radius * sin(angle)
        lon_offset = dx / 111139
        lat_offset = dy / 111139
        sector_points.append((lon + lon_offset, lat + lat_offset))
    sector_points.append((lon, lat))  # close the sector
    sector_poly = gpd.GeoSeries([Point(p) for p in sector_points]).union_all().convex_hull
    nearby = features_gdf[features_gdf.geometry.within(sector_poly)]
    return nearby

# Helper function to extract patch
def extract_patch(image, center_x, center_y, size=224):
    h, w = image.shape[:2]
    half_size = size // 2

    x1 = max(center_x - half_size, 0)
    y1 = max(center_y - half_size, 0)
    x2 = min(center_x + half_size, w)
    y2 = min(center_y + half_size, h)

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

os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

for idx, row in tqdm(gps_df.iterrows(), total=len(gps_df), desc="Processing Images"):
    image_path = os.path.join(IMAGE_DIR, row['image'])
    depth_path = os.path.join(DEPTH_DIR, row['image'])

    if not os.path.exists(image_path) or not os.path.exists(depth_path):
        continue

    img = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    img_h, img_w = img.shape[:2]
    depth_h, depth_w = depth.shape[:2]

    detections = yolo.predict(img, conf=0.2, classes=CLASS_IDS, verbose=False)
    if len(detections) == 0:
        continue

    vis_img = img.copy()

    if 1 < idx < len(gps_df) - 1:
        prev_coord = (float(gps_df.iloc[idx - 2]['latitude']), float(gps_df.iloc[idx - 2]['longitude']))
        next_coord = (float(gps_df.iloc[idx]['latitude']), float(gps_df.iloc[idx]['longitude']))
        heading = calculate_heading(prev_coord, next_coord)
    else:
        heading = None

    if heading is None:
        continue

    nearby_features = find_forward_facing_features((row['latitude'], row['longitude']), heading, RADIUS)

    for det in detections:
        for cls, box in zip(det.boxes.cls.cpu().numpy(), det.boxes.xywh.cpu().numpy()):
            class_id = int(cls)
            center_x_img, center_y_img = int(box[0]), int(box[1])
            width, height = int(box[2]), int(box[3])

            x1 = max(center_x_img - width // 2, 0)
            y1 = max(center_y_img - height // 2, 0)
            x2 = min(center_x_img + width // 2, img_w)
            y2 = min(center_y_img + height // 2, img_h)
            color = (0, 255, 0) if class_id == 4 else (255, 0, 0)
            label = CLASS_NAMES.get(class_id, 'unknown')
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            center_x = int(center_x_img * (depth_w / img_w))
            center_y = int(center_y_img * (depth_h / img_h))
            depth_value = depth[center_y, center_x]

            if nearby_features.empty:
                continue

            for _, feature in nearby_features.iterrows():
                if class_id == 4:
                    feature_id = feature['vine_id']
                elif class_id == 2:
                    feature_id = feature['row_post_id']
                else:
                    feature_id = feature.name

                class_dir = CLASS_NAMES.get(class_id, 'unknown')
                feature_dir = os.path.join(OUTPUT_DIR, class_dir, str(feature_id))
                os.makedirs(feature_dir, exist_ok=True)

                patch = extract_patch(img, center_x_img, center_y_img)
                output_filename = os.path.join(feature_dir, f"{row['image']}")
                cv2.imwrite(output_filename, patch)

    vis_output_path = os.path.join(YOLO_OUTPUT_DIR, row['image'])
    cv2.imwrite(vis_output_path, vis_img)

print("Extraction completed.")
