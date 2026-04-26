import cv2
import numpy as np
import open3d as o3d
import os

input_folder = "cropped_data"
output_folder = "cropped_data_masks"
os.makedirs(output_folder, exist_ok=True)

FX, FY = 500, 500
DEPTH_SCALE = 1.0
DEPTH_MIN = 1.0
DEPTH_MAX = 255.0
GROUND_ROW_FRAC = 0.4
MIN_GROUND_NORMAL_Y = 0.5
PLANE_DIST_THRESH = 1.0
OBSTACLE_HEIGHT_THRESH = 2.0

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(input_folder, filename)
    depth_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        continue
    if len(depth_raw.shape) == 3:
        depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

    depth = depth_raw.astype(np.float32) / DEPTH_SCALE
    h, w = depth.shape
    cx, cy = w / 2.0, h / 2.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid_depth = (depth > DEPTH_MIN) & (depth < DEPTH_MAX)

    z = depth
    x = (u - cx) * z / FX
    y = (v - cy) * z / FY

    ground_rows = v >= int(h * GROUND_ROW_FRAC)
    fit_mask = valid_depth & ground_rows
    pts_fit = np.stack([x[fit_mask], y[fit_mask], z[fit_mask]], axis=-1)

    if len(pts_fit) < 100:
        print(f"Skipping {filename}: not enough ground points")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_fit)
    pcd = pcd.voxel_down_sample(voxel_size=0.5)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) < 50:
        print(f"Skipping {filename}: too few points after filter")
        continue

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_THRESH,
        ransac_n=3,
        num_iterations=2000,
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # ── KEY FIX: your depth image is inverted (bright=far, dark=near is flipped)
    # Obstacles are CLOSER than ground → they have LOWER z → negative signed dist
    # So flip: we want normal pointing toward the camera (positive z component)
    if normal[2] < 0:
        normal = -normal
        d = -d

    print(f"{filename} | normal={normal.round(3)} | abs(ny)={abs(normal[1]):.3f}")

    if abs(normal[1]) < MIN_GROUND_NORMAL_Y:
        print(f"  REJECTED: bad plane normal")
        continue

    pts_all = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    signed_dist = (pts_all @ normal + d).reshape(h, w)
    signed_dist[~valid_depth] = 0.0

    print(f"  signed_dist min={signed_dist.min():.3f} max={signed_dist.max():.3f}")

    # ── Obstacles are BELOW the plane in signed dist (they're closer to camera)
    # So threshold on NEGATIVE signed distance
    obstacle_mask = (signed_dist < -OBSTACLE_HEIGHT_THRESH) & valid_depth

    print(f"  obstacle pixels: {obstacle_mask.sum()}")

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = obstacle_mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

    # Debug: visualize signed dist with obstacle highlighted
    sd_vis = signed_dist.copy()
    # Normalize so negative (obstacles) appear bright
    sd_neg = np.clip(-signed_dist, 0, None)
    sd_vis_norm = cv2.normalize(sd_neg, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_folder, f"DEBUG_{filename}"), sd_vis_norm.astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, filename), mask_u8)
    print(f"  Saved: {filename}")

print("Done.")