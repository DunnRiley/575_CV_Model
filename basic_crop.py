import os
import re
import cv2

input_folder = "simulated_data"
output_folder = "cropped_data"

os.makedirs(output_folder, exist_ok=True)

# --- Settings ---
bottom_fraction = 0.7
middle_fraction = 0.8

# --- Helper: extract timestamp from filename ---
def extract_timestamp(filename):
    # grabs the long number before .png
    match = re.search(r'(\d+\.\d+)', filename)
    return float(match.group(1)) if match else 0

# --- Get and sort files ---
files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
files.sort(key=extract_timestamp)

print(f"[INFO] Found {len(files)} images")

# --- Process images ---
for i, filename in enumerate(files):
    input_path = os.path.join(input_folder, filename)
    img = cv2.imread(input_path)

    if img is None:
        print(f"[WARN] Skipping {filename}")
        continue

    h, w = img.shape[:2]

    # Vertical crop (bottom)
    y_start = int(h * (1 - bottom_fraction))
    y_end = h

    # Horizontal crop (center)
    x_margin = int(w * (1 - middle_fraction) / 2)
    x_start = x_margin
    x_end = w - x_margin

    cropped = img[y_start:y_end, x_start:x_end]

    # New filename
    new_name = f"frame_{i:04d}.png"
    output_path = os.path.join(output_folder, new_name)

    cv2.imwrite(output_path, cropped)

print("[INFO] Done processing images")