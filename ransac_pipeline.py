import cv2
import numpy as np
import os

input_folder = "cropped_data"
output_folder = "cropped_data_masks"
os.makedirs(output_folder, exist_ok=True)

OBSTACLE_THRESH = 15.0  # tune this if needed

def jet_to_depth(color_img_bgr):
    hsv = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    val = hsv[:, :, 2]
    invalid = (val < 10)
    hue_clipped = np.clip(hue, 0, 120)
    depth = (hue_clipped / 120.0 * 255.0).astype(np.float32)
    depth[invalid] = 0.0
    return depth

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(input_folder, filename)
    color_img = cv2.imread(path, cv2.IMREAD_COLOR)
    if color_img is None:
        continue

    depth = jet_to_depth(color_img)
    h, w = depth.shape
    valid = (depth > 1.0) & (depth < 254.0)

    # Fit ground line from bottom rows
    bottom_rows = np.arange(int(h * 0.6), int(h * 0.92))
    row_median_depth = np.array([
        np.median(depth[r, :][valid[r, :]]) if valid[r, :].sum() > 10 else np.nan
        for r in bottom_rows
    ])

    good = ~np.isnan(row_median_depth)
    if good.sum() < 5:
        print(f"Skipping {filename}: not enough ground rows")
        continue

    coeffs = np.polyfit(bottom_rows[good], row_median_depth[good], 1)
    slope, intercept = coeffs
    all_rows = np.arange(h)
    expected_per_row = slope * all_rows + intercept
    expected = np.tile(expected_per_row[:, np.newaxis], (1, w))

    diff = depth.astype(np.float32) - expected.astype(np.float32)
    diff[~valid] = 0.0

    obstacle_mask = (diff > OBSTACLE_THRESH) & valid

    kernel = np.ones((7, 7), np.uint8)
    mask_u8 = obstacle_mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    cv2.imwrite(os.path.join(output_folder, filename), mask_u8)
    print(f"Processed: {filename}")

print("Done.")