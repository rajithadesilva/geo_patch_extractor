import os
import csv
import numpy as np
from scipy.interpolate import interp1d
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIG ---
image_folder = 'data/colour'
llh_file = 'data/gps/1_6.LLH'
output_csv = 'data/1_6_interpolated.csv'
map_folder = 'map'
os.makedirs(map_folder, exist_ok=True)

# Hardcoded matching
image_frame_numbers = [25, 315, 365, 547, 598, 675, 725, 880, 930, 1000, 1035, 1182, 1233, 1331, 1338, 1418]
llh_line_numbers = [0, 208, 236, 365, 397, 528, 558, 658, 686, 800, 822, 931, 965, 1068, 1091, 1204]

# --- Functions ---

def parse_llh(llh_path):
    gps_data = []
    with open(llh_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            timestamp = ' '.join(parts[:2])
            lat, lon, alt = map(float, parts[2:5])
            gps_data.append({'timestamp': timestamp, 'lat': lat, 'lon': lon, 'alt': alt})
    return gps_data

# Load GPS data
gps_data = parse_llh(llh_file)

# Prepare intervals
csv_rows = []
for idx in range(len(image_frame_numbers)-1):
    frame_start, frame_end = image_frame_numbers[idx], image_frame_numbers[idx+1]
    llh_start, llh_end = llh_line_numbers[idx], llh_line_numbers[idx+1]

    num_frames = frame_end - frame_start + 1
    num_llh_points = llh_end - llh_start + 1

    frame_range = list(range(frame_start, frame_end + 1))
    llh_segment = gps_data[llh_start:llh_end + 1]

    llh_indices = np.linspace(0, num_llh_points - 1, num_frames)

    lats = np.interp(llh_indices, np.arange(num_llh_points), [g['lat'] for g in llh_segment])
    lons = np.interp(llh_indices, np.arange(num_llh_points), [g['lon'] for g in llh_segment])
    alts = np.interp(llh_indices, np.arange(num_llh_points), [g['alt'] for g in llh_segment])

    for f, lat, lon, alt in zip(frame_range, lats, lons, alts):
        image_name = f"frame{f:04d}.jpg"
        csv_rows.append([image_name, lat, lon, alt])

# Save CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_file_name', 'latitude', 'longitude', 'altitude'])
    writer.writerows(csv_rows)

print(f"Interpolated GPS data saved to {output_csv}")

# --- Visualization ---
def visualize_map(csv_file, image_folder, map_folder):
    pairs = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pairs.append((row[0], float(row[1]), float(row[2])))

    all_coords = [(lat, lon) for _, lat, lon in pairs]

    for idx, (img_file, lat, lon) in enumerate(tqdm(pairs, desc="Generating maps")):
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: {img_path} not found, skipping.")
            continue

        fig, ax = plt.subplots(figsize=(3,3))
        ax.plot(*zip(*[(lon, lat) for lat, lon in all_coords]), color='gray', linewidth=1)
        ax.plot(*zip(*[(lon, lat) for lat, lon in all_coords[:idx+1]]), color='red', linewidth=2)
        ax.axis('off')
        plt.tight_layout()

        tmp_map = os.path.join(map_folder, 'tmp.png')
        plt.savefig(tmp_map, bbox_inches='tight', pad_inches=0)
        plt.close()

        map_img = cv2.imread(tmp_map)
        h, w = int(img.shape[0]*0.25), int(img.shape[1]*0.25)
        map_img = cv2.resize(map_img, (w, h))
        img[-h:, -w:] = map_img

        cv2.imwrite(os.path.join(map_folder, f'map_{img_file}'), img)
        os.remove(tmp_map)

print("Saving visualized maps...")
visualize_map(output_csv, image_folder, map_folder)
print(f"Maps saved in {map_folder}")

