"""
fix_masks_and_images.py
-----------------------
Two fixes in one script:

  1. MASKS (from og_masks/) — flood-fills the top invalid white blob with black.
     Seeds the fill from all corners of the top edge so slanted horizons
     are handled correctly regardless of shape.
     Saves cleaned masks to data/masks/

  2. IMAGES (from og_images/) — applies a conservative crop that only removes
     the RViz window chrome (title bar, borders) without touching depth data.
     Saves cropped images to data/images/

Usage:
  python3 fix_masks_and_images.py \
      --og_mask_dir  og_masks  \
      --og_image_dir og_images \
      --out_mask_dir data/masks \
      --out_image_dir data/images \
      [--img_top 25] [--img_x1 47] [--img_x2 858] [--img_y2 648]
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


# Mask fix: flood-fill top white blob
def fill_top_blob(mask: np.ndarray) -> np.ndarray:
    """
    Flood-fill the top invalid white region with black.
    Seeds from every pixel along the top row that is white,
    so slanted and irregular horizons are fully covered.
    """
    h, w = mask.shape
    canvas = mask.copy()

    # Seed from entire top row — catches any slant
    for x in range(w):
        if canvas[0, x] > 127:
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(canvas, ff_mask, (x, 0), 0,
                          loDiff=50, upDiff=50)

    # Also seed from top corners in case top row starts black but corners are white
    for seed in [(0, 0), (w-1, 0)]:
        if canvas[seed[1], seed[0]] > 127:
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(canvas, ff_mask, seed, 0,
                          loDiff=50, upDiff=50)

    return canvas


def process_mask(src_path: Path, dst_path: Path, visualize: bool, vis_dir: Path):
    mask = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [WARN] Cannot read {src_path.name}")
        return False

    before = np.sum(mask > 127)
    cleaned = fill_top_blob(mask)
    after = np.sum(cleaned > 127)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), cleaned)

    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
        # Side by side: original | cleaned
        orig_rgb    = cv2.cvtColor(mask,    cv2.COLOR_GRAY2BGR)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        divider = np.zeros((mask.shape[0], 4, 3), dtype=np.uint8)
        divider[:] = (80, 80, 80)
        side_by_side = np.hstack([orig_rgb, divider, cleaned_rgb])
        cv2.putText(side_by_side, "BEFORE", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(side_by_side, "AFTER", (mask.shape[1] + 8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(vis_dir / src_path.name), side_by_side)

    return before, after


# Image fix: conservative chrome-only crop
def process_image(src_path: Path, dst_path: Path,
                  img_top, img_y2, img_x1, img_x2):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  [WARN] Cannot read {src_path.name}")
        return False

    h, w = img.shape[:2]

    # Clamp crop coords to actual image size
    y1 = min(img_top, h); y2 = min(img_y2, h)
    x1 = min(img_x1,  w); x2 = min(img_x2, w)

    crop = img[y1:y2, x1:x2]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), crop)
    return True


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--og_mask_dir",   default="og_masks",
                        help="Original uncropped masks")
    parser.add_argument("--og_image_dir",  default="og_images",
                        help="Original uncropped screenshots")
    parser.add_argument("--out_mask_dir",  default="data/masks",
                        help="Output cleaned masks")
    parser.add_argument("--out_image_dir", default="data/images",
                        help="Output cropped images")
    # Conservative image crop — just removes RViz chrome, keeps all depth data
    parser.add_argument("--img_top", type=int, default=25,
                        help="Top crop row for images (default 25 = just below title bar)")
    parser.add_argument("--img_y2",  type=int, default=648,
                        help="Bottom crop row for images (default 648)")
    parser.add_argument("--img_x1",  type=int, default=47,
                        help="Left crop col for images (default 47)")
    parser.add_argument("--img_x2",  type=int, default=858,
                        help="Right crop col for images (default 858)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save before/after mask comparisons to visualizations_fill/")
    args = parser.parse_args()

    og_mask_dir  = Path(args.og_mask_dir)
    og_image_dir = Path(args.og_image_dir)
    out_mask_dir = Path(args.out_mask_dir)
    out_img_dir  = Path(args.out_image_dir)
    vis_dir      = Path("visualizations_fill")

    mask_files = sorted(og_mask_dir.glob("*.png"))
    img_files  = sorted(og_image_dir.glob("*.png"))

    if not mask_files:
        sys.exit(f"No .png files found in {og_mask_dir}")

    print(f"Processing {len(mask_files)} masks and {len(img_files)} images...\n")

    #  Masks
    print(" Flood-filling masks ")
    for mask_path in mask_files:
        result = process_mask(mask_path,
                              out_mask_dir / mask_path.name,
                              args.visualize, vis_dir)
        if result:
            before, after = result
            removed = before - after
            print(f"  {mask_path.name}  {before} → {after} white px  (-{removed})")

    #  Images 
    print(f"\n Cropping images [{args.img_top}:{args.img_y2}, {args.img_x1}:{args.img_x2}] ")
    done = 0
    for img_path in img_files:
        ok = process_image(img_path,
                           out_img_dir / img_path.name,
                           args.img_top, args.img_y2,
                           args.img_x1, args.img_x2)
        if ok:
            done += 1

    print(f"  {done} images cropped.")
    print(f"\n Done")
    print(f"  Masks  → {out_mask_dir}/")
    print(f"  Images → {out_img_dir}/")
    if args.visualize:
        print(f"  Mask previews → {vis_dir}/")
    print(f"\nNext: run mask_to_yolo.py to regenerate labels.")


if __name__ == "__main__":
    main()