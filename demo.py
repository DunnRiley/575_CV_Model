
"""
demo.py
-------
Runs your trained YOLOv8 terrain detector on saved image frames and outputs:
  1. Annotated images (RGB with detections | original mask) side by side
  2. A stitched video from those frames

Usage:
  python3 demo.py \
      --weights  best.pt \
      --image_dir data_demo/images \
      --mask_dir  data_demp/masks \
      --output_dir demo_output \
      [--conf 0.3] \
      [--fps 10]

Output:
  demo_output/
    frames/       <- individual side-by-side annotated images
    demo.mp4      <- video stitched from all frames
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not found. Run: pip install ultralytics")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colors per class  (BGR)
CLASS_COLORS = {
    "hole":  (0,   0,   255),   # red
    "mound": (0,   200,  0 ),   # green
}
DEFAULT_COLOR = (255, 165, 0)   # orange fallback


def draw_detections(image: np.ndarray, results) -> np.ndarray:
    """Draw bounding boxes + labels + confidence on a copy of image."""
    out = image.copy()
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        cv2.putText(out, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return out

    names = results[0].names   # {0: 'hole', 1: 'mound'}

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = names[cls]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        txt   = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Detection count overlay
    n_holes  = sum(1 for b in boxes if results[0].names[int(b.cls[0])] == "hole")
    n_mounds = sum(1 for b in boxes if results[0].names[int(b.cls[0])] == "mound")
    summary  = f"Holes: {n_holes}  Mounds: {n_mounds}"
    cv2.putText(out, summary, (10, out.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out


def mask_to_rgb(mask_path: Path, target_size: tuple) -> np.ndarray:
    """Load binary mask and convert to a coloured RGB panel."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        # Return a blank panel if mask missing
        blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(blank, "No mask", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        return blank

    mask = cv2.resize(mask, target_size)
    # White regions → light blue tint so it reads nicely next to RGB
    coloured = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    coloured[mask > 127] = (200, 180, 60)    # gold  = object regions
    coloured[mask == 0]  = (30,  30,  30)    # dark  = ground / removed
    return coloured


def add_panel_label(panel: np.ndarray, text: str) -> np.ndarray:
    """Add a small title bar at the top of a panel."""
    bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
    bar[:] = (50, 50, 50)
    cv2.putText(bar, text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return np.vstack([bar, panel])


def make_side_by_side(rgb_det: np.ndarray, mask_panel: np.ndarray) -> np.ndarray:
    """Stack detection panel and mask panel horizontally with labels."""
    h = max(rgb_det.shape[0], mask_panel.shape[0])

    def pad(img):
        if img.shape[0] < h:
            pad_h = h - img.shape[0]
            img = np.vstack([img, np.zeros((pad_h, img.shape[1], 3), dtype=np.uint8)])
        return img

    left  = add_panel_label(pad(rgb_det),    "YOLOv8 Detections")
    right = add_panel_label(pad(mask_panel), "RANSAC Mask")
    divider = np.zeros((left.shape[0], 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    return np.hstack([left, divider, right])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Terrain detector demo")
    parser.add_argument("--weights",    required=True, help="Path to best.pt")
    parser.add_argument("--image_dir",  required=True, help="Directory of RGB frames")
    parser.add_argument("--mask_dir",   required=True, help="Directory of binary masks")
    parser.add_argument("--output_dir", default="demo_output")
    parser.add_argument("--conf",  type=float, default=0.3,
                        help="Confidence threshold (default 0.3)")
    parser.add_argument("--fps",   type=int,   default=10,
                        help="FPS for output video (default 10)")
    parser.add_argument("--device", default=None,
                        help="Inference device: cpu | 0  (default: auto)")
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    mask_dir   = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # ── Gather frames ─────────────────────────────────────────────────────────
    image_files = sorted(f for f in image_dir.glob("*")
                         if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_files:
        sys.exit(f"No images found in {image_dir}")

    print(f"Running inference on {len(image_files)} frames...\n")

    frame_paths = []
    video_writer = None

    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Cannot read {img_path.name}, skipping.")
            continue

        img_h, img_w = image.shape[:2]

        # ── Inference ─────────────────────────────────────────────────────────
        predict_kwargs = dict(conf=args.conf, verbose=False)
        if args.device:
            predict_kwargs["device"] = args.device
        results = model.predict(image, **predict_kwargs)

        # ── Build panels ──────────────────────────────────────────────────────
        det_panel  = draw_detections(image, results)
        mask_path  = mask_dir / img_path.name
        mask_panel = mask_to_rgb(mask_path, (img_w, img_h))
        composite  = make_side_by_side(det_panel, mask_panel)

        # ── Save frame ────────────────────────────────────────────────────────
        out_path = frames_dir / img_path.name
        cv2.imwrite(str(out_path), composite)
        frame_paths.append(out_path)

        # ── Init video writer on first frame ──────────────────────────────────
        if video_writer is None:
            h, w = composite.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = output_dir / "demo.mp4"
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))

        video_writer.write(composite)

        # Count detections for logging
        n = len(results[0].boxes) if results[0].boxes else 0
        print(f"  [{i+1:>4}/{len(image_files)}] {img_path.name}  →  {n} detections")

    if video_writer:
        video_writer.release()

    print(f"\n── Demo complete ─────────────────────────────────────────────")
    print(f"  Annotated frames : {frames_dir}/")
    print(f"  Video            : {output_dir}/demo.mp4")
    print(f"  Total frames     : {len(frame_paths)}")
    print("──────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()