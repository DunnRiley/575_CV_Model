import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("scikit-learn not found. Run: pip install scikit-learn")


# Clustering

def load_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def dbscan_clusters(mask, eps, min_samples, downsample):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    coords = np.column_stack([xs, ys])
    coords_ds = coords[::downsample]
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(coords_ds)
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        clusters.append(coords_ds[labels == label])
    return clusters


def cluster_to_bbox(cluster):
    x_min, y_min = cluster.min(axis=0)
    x_max, y_max = cluster.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


# Box merging

def box_iou(b1, b2):
    """Intersection over Union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def box_distance(b1, b2):
    """
    Centre-to-centre distance between two boxes normalised by their avg diagonal.
    0 = same centre, 1 = ~1 diagonal apart.
    """
    cx1 = (b1[0]+b1[2])/2; cy1 = (b1[1]+b1[3])/2
    cx2 = (b2[0]+b2[2])/2; cy2 = (b2[1]+b2[3])/2
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    diag1 = np.sqrt((b1[2]-b1[0])**2 + (b1[3]-b1[1])**2)
    diag2 = np.sqrt((b2[2]-b2[0])**2 + (b2[3]-b2[1])**2)
    avg_diag = (diag1 + diag2) / 2
    return dist / avg_diag if avg_diag > 0 else 0.0


def merge_boxes(boxes, merge_iou=0.1, merge_dist=0.5):
    """
    Greedily merge boxes that either overlap (IoU > merge_iou) or are
    close together (normalised centre distance < merge_dist).
    Returns list of merged (x1,y1,x2,y2) boxes.
    """
    if not boxes:
        return []

    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        result = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            current = list(boxes[i])
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                iou  = box_iou(current, boxes[j])
                dist = box_distance(current, boxes[j])
                if iou > merge_iou or dist < merge_dist:
                    # Merge: take the union bounding box
                    current[0] = min(current[0], boxes[j][0])
                    current[1] = min(current[1], boxes[j][1])
                    current[2] = max(current[2], boxes[j][2])
                    current[3] = max(current[3], boxes[j][3])
                    used[j] = True
                    merged = True
            result.append(tuple(current))
            used[i] = True
        boxes = result

    return boxes


# Classification & formatting

def classify_by_area(cluster, area_threshold):
    """
    Large clusters = hole (0) — holes are big flat regions in the mask.
    Small clusters = mound (1) — mounds are smaller raised blobs.
    """
    return 0 if len(cluster) >= area_threshold else 1


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


# Per-image processing

def process_one(image_path, mask_path, label_path,
                eps, min_samples, downsample, min_blob_area,
                area_threshold, min_box_wh, merge_iou, merge_dist,
                visualize, vis_dir):

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [WARN] Cannot read image {image_path}, skipping.")
        return 0

    img_h, img_w = image.shape[:2]
    mask = load_mask(mask_path)

    # Resize mask to match image if needed
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    clusters = dbscan_clusters(mask, eps=eps, min_samples=min_samples,
                               downsample=downsample)

    #  Per-cluster: filter tiny blobs, classify, get bbox
    # Group bboxes by class for per-class merging
    class_boxes = {0: [], 1: []}

    for cluster in clusters:
        if len(cluster) < min_blob_area:
            continue
        cls = classify_by_area(cluster, area_threshold)
        bbox = cluster_to_bbox(cluster)
        class_boxes[cls].append(bbox)

    #  Merge boxes within each class 
    lines = []
    vis_image = image.copy() if visualize else None

    CLASS_NAMES  = {0: "hole",  1: "mound"}
    CLASS_COLORS = {0: (0, 0, 255), 1: (0, 200, 0)}   # red=hole, green=mound

    for cls, boxes in class_boxes.items():
        merged = merge_boxes(boxes, merge_iou=merge_iou, merge_dist=merge_dist)

        for (x_min, y_min, x_max, y_max) in merged:
            # Filter boxes that are too small relative to image size
            box_w = (x_max - x_min) / img_w
            box_h = (y_max - y_min) / img_h
            if box_w < min_box_wh or box_h < min_box_wh:
                continue

            cx, cy, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if visualize:
                color     = CLASS_COLORS[cls]
                label_txt = CLASS_NAMES[cls]
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(vis_image, label_txt, (x_min, max(y_min - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    if visualize and vis_image is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / image_path.name), vis_image)

    return len(lines)


# Main

def main():
    parser = argparse.ArgumentParser(description="DBSCAN mask → YOLO labels")
    parser.add_argument("--image_dir",      required=True)
    parser.add_argument("--mask_dir",       required=True)
    parser.add_argument("--label_dir",      required=True)
    parser.add_argument("--downsample",     type=int,   default=5,
                        help="Use every Nth white pixel for DBSCAN (default 5)")
    parser.add_argument("--eps",            type=float, default=20,
                        help="DBSCAN neighbourhood radius (default 20, increase to merge nearby clusters)")
    parser.add_argument("--min_samples",    type=int,   default=5,
                        help="DBSCAN min points to form a cluster (default 5)")
    parser.add_argument("--min_blob_area",  type=int,   default=30,
                        help="Min cluster size in downsampled pixels (default 30)")
    parser.add_argument("--min_box_wh",     type=float, default=0.03,
                        help="Min box width AND height as fraction of image (default 0.03 = 3%%)")
    parser.add_argument("--merge_iou",      type=float, default=0.1,
                        help="Merge boxes with IoU above this (default 0.1)")
    parser.add_argument("--merge_dist",     type=float, default=0.5,
                        help="Merge boxes whose centres are within this many diagonals (default 0.5)")
    parser.add_argument("--area_threshold", type=int,   default=300,
                        help="Downsampled pixel count: >= this = hole (0), else mound (1). Default 300")
    parser.add_argument("--visualize",      action="store_true",
                        help="Save annotated images to visualizations/")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir  = Path(args.mask_dir)
    label_dir = Path(args.label_dir)
    vis_dir   = Path("visualizations")

    image_files = sorted(f for f in image_dir.glob("*")
                         if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_files:
        sys.exit(f"No images found in {image_dir}")

    print(f"Found {len(image_files)} images. Processing...\n")

    total_boxes = 0
    for img_path in image_files:
        mask_path  = mask_dir  / img_path.name
        label_path = label_dir / img_path.with_suffix(".txt").name

        if not mask_path.exists():
            print(f"  [WARN] No mask for {img_path.name}, skipping.")
            continue

        n = process_one(img_path, mask_path, label_path,
                        args.eps, args.min_samples, args.downsample,
                        args.min_blob_area, args.area_threshold,
                        args.min_box_wh, args.merge_iou, args.merge_dist,
                        args.visualize, vis_dir)

        print(f"  {img_path.name}  →  {n} boxes")
        total_boxes += n

    print(f"\nDone. {len(image_files)} images processed, {total_boxes} total boxes.")
    print(f"Labels saved to: {label_dir}")
    if args.visualize:
        print(f"Visualizations saved to: {vis_dir}/")


if __name__ == "__main__":
    main()