import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN

input_folder = "masks"
output_folder = "clustered"

os.makedirs(output_folder, exist_ok=True)

# noise filtering parameters
MIN_CLUSTER_SIZE = 5
DBSCAN_EPS = 8
DBSCAN_MIN_SAMPLES = 30

for filename in os.listdir(input_folder):

    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(input_folder, filename)

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        continue

    # ----------------------------
    # 1. MORPHOLOGICAL NOISE CLEAN
    # ----------------------------

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ----------------------------
    # 2. REMOVE SMALL BLOBS
    # ----------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_CLUSTER_SIZE:
            clean_mask[labels == i] = 255

    # ----------------------------
    # 3. EXTRACT OBSTACLE POINTS
    # ----------------------------

    points = np.column_stack(np.where(clean_mask > 0))

    if len(points) < DBSCAN_MIN_SAMPLES:
        print("Skipping", filename)
        continue

    # ----------------------------
    # 4. CLUSTER WITH DBSCAN
    # ----------------------------

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)

    labels = clustering.labels_

    # convert mask to color image for drawing
    output = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

    unique_labels = set(labels)

    rock_count = 0

    for label in unique_labels:

        if label == -1:
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < MIN_CLUSTER_SIZE:
            continue

        rock_count += 1

        y_coords = cluster_points[:,0]
        x_coords = cluster_points[:,1]

        x1 = np.min(x_coords)
        x2 = np.max(x_coords)

        y1 = np.min(y_coords)
        y2 = np.max(y_coords)

        # draw bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        # label object
        cv2.putText(
            output,
            f"obj {rock_count}",
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0,255,0),
            1
        )

    # ----------------------------
    # 5. SAVE RESULT
    # ----------------------------

    save_path = os.path.join(output_folder, filename)

    cv2.imwrite(save_path, output)

    print(f"{filename} -> {rock_count} objects detected")

print("Processing complete")