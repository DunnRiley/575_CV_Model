"""
mask_to_yolo.py
---------------
Takes paired (image, mask) files and uses DBSCAN clustering on the white
mask pixels to generate YOLO-format bounding box labels.

Classes:
  0 = hole / depression
  1 = terrain obstacle / mound

Usage:
  python3 mask_to_yolo.py \
      --image_dir  data/images \
      --mask_dir   data/masks  \
      --label_dir  data/labels \
      [--depth_dir data/depth] \
      [--downsample 5]         \
      [--eps 15] [--min_samples 5] \
      [--min_blob_area 20] \
      [--area_threshold 500] \
      [--visualize]
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    sys.exit("scikit-learn not found. Run:  pip install scikit-learn")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def dbscan_clusters(mask: np.ndarray, eps: float, min_samples: int, downsample: int):
    """
    Run DBSCAN on white pixels of the mask.
    Downsamples by taking every Nth pixel to keep clustering fast on large masks.
    Returns list of pixel coordinate arrays (downsampled resolution), one per cluster.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    coords = np.column_stack([xs, ys])      # (N, 2)
    coords_ds = coords[::downsample]        # downsample for speed

    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(coords_ds)

    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        clusters.append(coords_ds[labels == label])

    return clusters


def cluster_to_bbox(cluster: np.ndarray):
    x_min, y_min = cluster.min(axis=0)
    x_max, y_max = cluster.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return cx, cy, w, h


def classify_with_depth(cluster, depth_gray, img_h, img_w, margin=10):
    """
    Strategy A: compare median depth inside bbox vs surrounding border ring.
    Brighter inside = closer = mound (1)
    Darker inside   = farther = hole  (0)
    """
    x_min, y_min, x_max, y_max = cluster_to_bbox(cluster)
    x1 = max(x_min, 0);       y1 = max(y_min, 0)
    x2 = min(x_max, img_w-1); y2 = min(y_max, img_h-1)

    roi = depth_gray[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return 1

    median_inside = np.median(roi)

    bx1 = max(x1-margin, 0);       by1 = max(y1-margin, 0)
    bx2 = min(x2+margin, img_w-1); by2 = min(y2+margin, img_h-1)
    border_roi = depth_gray[by1:by2+1, bx1:bx2+1].copy()
    inner_h = min(roi.shape[0], border_roi.shape[0] - margin)
    inner_w = min(roi.shape[1], border_roi.shape[1] - margin)
    border_roi[margin:margin+inner_h, margin:margin+inner_w] = 0
    border_vals = border_roi[border_roi > 0]
    median_border = np.median(border_vals) if len(border_vals) else median_inside

    return 1 if median_inside > median_border else 0


def classify_by_area(cluster, area_threshold):
    """Strategy B (no depth): large blobs = mound (1), small = hole (0)."""
    return 1 if len(cluster) >= area_threshold else 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ─────────────────────────────────────────────────────────────────────────────

def process_one(image_path, mask_path, label_path, depth_path,
                eps, min_samples, downsample, min_blob_area,
                area_threshold, visualize, vis_dir):

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [WARN] Cannot read image {image_path}, skipping.")
        return 0

    img_h, img_w = image.shape[:2]
    mask = load_mask(mask_path)

    depth_gray = None
    if depth_path and depth_path.exists():
        d = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if d is not None:
            depth_gray = cv2.resize(d, (img_w, img_h))

    clusters = dbscan_clusters(mask, eps=eps, min_samples=min_samples,
                               downsample=downsample)

    lines = []
    vis_image = image.copy() if visualize else None

    for cluster in clusters:
        if len(cluster) < min_blob_area:
            continue

        x_min, y_min, x_max, y_max = cluster_to_bbox(cluster)
        cx, cy, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        if depth_gray is not None:
            cls = classify_with_depth(cluster, depth_gray, img_h, img_w)
        else:
            cls = classify_by_area(cluster, area_threshold)

        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if visualize:
            color     = (0, 255, 0) if cls == 1 else (0, 0, 255)  # green=mound, red=hole
            label_txt = "mound"     if cls == 1 else "hole"
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(vis_image, label_txt, (x_min, max(y_min - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    if visualize and vis_image is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / image_path.name), vis_image)

    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DBSCAN mask → YOLO labels")
    parser.add_argument("--image_dir",      required=True,  help="Directory of RGB images")
    parser.add_argument("--mask_dir",       required=True,  help="Directory of binary masks")
    parser.add_argument("--label_dir",      required=True,  help="Output directory for YOLO .txt labels")
    parser.add_argument("--depth_dir",      default=None,   help="(Optional) Depth image directory")
    parser.add_argument("--downsample",     type=int,   default=5,
                        help="Use every Nth white pixel for DBSCAN (default 5, increase if slow)")
    parser.add_argument("--eps",            type=float, default=15,
                        help="DBSCAN neighbourhood radius in downsampled pixels (default 15)")
    parser.add_argument("--min_samples",    type=int,   default=5,
                        help="DBSCAN min points to form a cluster (default 5)")
    parser.add_argument("--min_blob_area",  type=int,   default=20,
                        help="Min cluster size in downsampled pixels to keep (default 20)")
    parser.add_argument("--area_threshold", type=int,   default=500,
                        help="(No-depth fallback) downsampled pixel count >= this → mound, else hole")
    parser.add_argument("--visualize",      action="store_true",
                        help="Save annotated images to visualizations/ folder")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir  = Path(args.mask_dir)
    label_dir = Path(args.label_dir)
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
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
        depth_path = (depth_dir / img_path.name) if depth_dir else None

        if not mask_path.exists():
            print(f"  [WARN] No mask for {img_path.name}, skipping.")
            continue

        n = process_one(img_path, mask_path, label_path, depth_path,
                        args.eps, args.min_samples, args.downsample,
                        args.min_blob_area, args.area_threshold,
                        args.visualize, vis_dir)

        print(f"  {img_path.name}  →  {n} boxes")
        total_boxes += n

    print(f"\nDone. {len(image_files)} images processed, {total_boxes} total boxes.")
    print(f"Labels saved to: {label_dir}")
    if args.visualize:
        print(f"Visualizations saved to: {vis_dir}/")


if __name__ == "__main__":
    main()