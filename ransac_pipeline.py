import cv2
import numpy as np
import open3d as o3d
import os

input_folder = "GoodData"
output_folder = "GoodData_masks"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(input_folder, filename)

    # Load 16-bit depth image
    depth = cv2.imread(path)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        continue

    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    h, w = depth.shape

    # Convert mm → meters
    depth = depth.astype(np.float32) / 1000.0

    h, w = depth.shape

    # Downsample for speed
    # depth = depth[::2, ::2]

    h, w = depth.shape

    # Camera intrinsics (approximate)
    fx = fy = 500
    cx = w / 2
    cy = h / 2

    # Pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Build point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Remove invalid depth
    valid = (points[:,2] > 0.1) & (points[:,2] < 5.0)
    points = points[valid]

    if len(points) < 100:
        print("Skipping", filename)
        continue

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RANSAC plane detection
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )

    a, b, c, d = plane_model

    # Compute distance of every pixel to plane
    points_full = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    dist = np.abs(points_full @ np.array([a, b, c]) + d) / np.sqrt(a*a + b*b + c*c)

    height_map = dist.reshape(h, w)

    # Obstacle mask
    mask = height_map > 0.02
    mask_img = (mask.astype(np.uint8)) * 255

    output_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_path, mask_img)

    print("Processed", filename)

print("Finished batch processing")