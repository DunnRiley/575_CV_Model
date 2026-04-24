"""
recrop_masks.py
---------------
Recrops original masks from /og_masks by:
  1. Detecting where the top invalid white blob ends (the dark-red depth region
     that RANSAC incorrectly flags as non-ground)
  2. Cropping from just below that blob to the bottom of the mask
  3. Also crops the matching depth images in data/images to the same region

The crop row is found dynamically per image so frames where the blob is
larger or smaller are handled correctly.

Usage:
  python3 recrop_masks.py \
      --og_mask_dir  og_masks \
      --image_dir    data/images \
      --out_mask_dir data/masks \
      --out_image_dir data/images_cropped \
      [--top_pct_threshold 0.15] \
      [--visualize]
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


def find_top_blob_end(mask: np.ndarray, pct_threshold: float) -> int:
    """
    Scan rows top-to-bottom. The top invalid blob is the large white region
    at the top of the mask (invalid depth flagged as non-ground).

    Returns the first row BELOW the top blob where white pixel percentage
    drops below pct_threshold. This becomes the new top of the crop.

    If no blob is found (mask is mostly black at top), returns 0.
    """
    h, w = mask.shape
    in_blob = False

    for row in range(h):
        white_pct = np.sum(mask[row] > 127) / w
        if white_pct > pct_threshold:
            in_blob = True
        elif in_blob and white_pct < pct_threshold:
            # Add a small buffer below the blob edge
            return max(0, row - 2)

    # If we never exited the blob, crop at 60% of image height as fallback
    return int(h * 0.6)


def find_crop_for_image(screenshot: np.ndarray, og_mask: np.ndarray, top_row_mask: int):
    """
    Map the mask crop row back to screenshot pixel coordinates.
    og_mask and screenshot may be different resolutions.
    Returns (img_top_row, img_y2, img_x1, img_x2) for the screenshot crop.
    """
    mask_h, mask_w = og_mask.shape[:2]
    img_h,  img_w  = screenshot.shape[:2]

    scale_y = img_h / mask_h
    scale_x = img_w / mask_w

    img_top = int(top_row_mask * scale_y)
    # Use full width minus any border (assume small border ~2% each side)
    border_x = int(img_w * 0.02)
    return img_top, img_h, border_x, img_w - border_x


def process_pair(img_path, mask_src_path, out_mask_path, out_img_path,
                 pct_threshold, visualize, vis_dir):

    og_mask = cv2.imread(str(mask_src_path), cv2.IMREAD_GRAYSCALE)
    if og_mask is None:
        print(f"  [WARN] Cannot read mask {mask_src_path.name}, skipping.")
        return False

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"  [WARN] Cannot read image {img_path.name}, skipping.")
        return False

    # ── Find where top invalid blob ends in the mask ──────────────────────────
    top_row_mask = find_top_blob_end(og_mask, pct_threshold)

    # ── Crop the mask ─────────────────────────────────────────────────────────
    mask_crop = og_mask[top_row_mask:, :]

    # ── Map to image coordinates and crop image ───────────────────────────────
    img_top, img_y2, img_x1, img_x2 = find_crop_for_image(image, og_mask, top_row_mask)
    image_crop = image[img_top:img_y2, img_x1:img_x2]

    # ── Resize mask to match image crop size ──────────────────────────────────
    # Keep mask at its own resolution — YOLO labels are normalised anyway
    # But ensure consistent aspect ratio
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_img_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_mask_path), mask_crop)
    cv2.imwrite(str(out_img_path),  image_crop)

    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
        # Draw crop line on original mask for inspection
        vis = cv2.cvtColor(og_mask, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (0, top_row_mask), (vis.shape[1], top_row_mask), (0, 255, 0), 2)
        cv2.putText(vis, f"crop y={top_row_mask}", (4, top_row_mask - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imwrite(str(vis_dir / mask_src_path.name), vis)

    return top_row_mask


def main():
    parser = argparse.ArgumentParser(description="Recrop original masks by removing top invalid blob")
    parser.add_argument("--og_mask_dir",    default="og_masks",
                        help="Directory containing original uncropped masks")
    parser.add_argument("--image_dir",      default="data/images",
                        help="Directory containing matching screenshot images")
    parser.add_argument("--out_mask_dir",   default="data/masks",
                        help="Output directory for cropped masks (overwrites existing)")
    parser.add_argument("--out_image_dir",  default="data/images",
                        help="Output directory for cropped images (overwrites existing)")
    parser.add_argument("--top_pct_threshold", type=float, default=0.15,
                        help="Row is considered 'in top blob' if white pixels > this fraction (default 0.15)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save annotated masks showing the detected crop line to visualizations/")
    args = parser.parse_args()

    og_mask_dir   = Path(args.og_mask_dir)
    image_dir     = Path(args.image_dir)
    out_mask_dir  = Path(args.out_mask_dir)
    out_img_dir   = Path(args.out_image_dir)
    vis_dir       = Path("visualizations_crop")

    mask_files = sorted(og_mask_dir.glob("*.png"))
    if not mask_files:
        sys.exit(f"No .png files found in {og_mask_dir}")

    print(f"Found {len(mask_files)} masks in {og_mask_dir}\n")

    crop_rows = []
    skipped   = 0

    for mask_path in mask_files:
        img_path     = image_dir   / mask_path.name
        out_mask     = out_mask_dir / mask_path.name
        out_img      = out_img_dir  / mask_path.name

        if not img_path.exists():
            print(f"  [WARN] No image for {mask_path.name}, skipping.")
            skipped += 1
            continue

        row = process_pair(img_path, mask_path, out_mask, out_img,
                           args.top_pct_threshold, args.visualize, vis_dir)
        if row is not False:
            crop_rows.append(row)
            print(f"  {mask_path.name}  crop_row={row}")

    print(f"\n── Done ─────────────────────────────────────────────────────")
    print(f"  Processed : {len(crop_rows)} masks")
    print(f"  Skipped   : {skipped}")
    if crop_rows:
        print(f"  Avg crop row : {np.mean(crop_rows):.1f}  "
              f"(min={min(crop_rows)}, max={max(crop_rows)})")
    print(f"  Masks  → {out_mask_dir}/")
    print(f"  Images → {out_img_dir}/")
    if args.visualize:
        print(f"  Crop preview → {vis_dir}/")
    print(f"\nNext: run mask_to_yolo.py on the new cropped files.")


if __name__ == "__main__":
    main()