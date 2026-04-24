import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# =====================================
# 1. LOAD IMAGE
# =====================================

image_path = "rgbd.png"
img = cv2.imread(image_path)

# convert RGB -> grayscale
if len(img.shape) == 3:
    depth = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if img is None:
    raise Exception("Image failed to load")

h, w, _ = img.shape
print("Loaded image:", img.shape)

# =====================================
# 2. CROP DEPTH PORTION (TOP HALF)
# =====================================

depth_vis = img[:h//2]

# remove window borders (optional)
depth_vis = depth_vis[40:-20, 20:-20]

# =====================================
# 3. CONVERT TO FAKE DEPTH
# =====================================

gray = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2GRAY)

# normalize fake depth
depth = gray.astype(np.float32) / 255.0

# downsample for speed
depth = depth[::2, ::2]

h, w = depth.shape

# =====================================
# 4. BUILD POINT CLOUD
# =====================================

# fake intrinsics (just for testing)
fx = fy = 500
cx = w / 2
cy = h / 2

u, v = np.meshgrid(np.arange(w), np.arange(h))

z = depth
x = (u - cx) * z / fx
y = (v - cy) * z / fy

points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

# remove invalid points
valid = points[:,2] > 0
points = points[valid]

print("Point cloud size:", len(points))

# =====================================
# 5. RANSAC PLANE FIT
# =====================================

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

a,b,c,d = plane_model
print("Plane:", plane_model)

# =====================================
# 6. COMPUTE HEIGHT ABOVE GROUND (HAG)
# =====================================

points_full = np.stack((x,y,z), axis=-1).reshape(-1,3)

dist = np.abs(points_full @ np.array([a,b,c]) + d) / np.sqrt(a*a+b*b+c*c)

height_map = dist.reshape(h,w)

# =====================================
# 7. OBSTACLE DETECTION
# =====================================

threshold = 0.02
mask = height_map > threshold

mask_img = (mask.astype(np.uint8))*255

cv2.imwrite("rock_mask.png", mask_img)

print("Mask saved as rock_mask.png")

# =====================================
# 8. VISUALIZE HEIGHT MAP
# =====================================

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Fake Depth")
plt.imshow(depth, cmap='gray')

plt.subplot(1,2,2)
plt.title("Height Above Ground")
plt.imshow(height_map, cmap='inferno')

plt.show()

# =====================================
# 9. VISUALIZE POINT CLOUD
# =====================================

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

# ground = green
inlier_cloud.paint_uniform_color([0,1,0])

# obstacles = red
outlier_cloud.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])