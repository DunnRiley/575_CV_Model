import argparse
import subprocess
import sys


def ensure_ultralytics():
    try:
        import ultralytics  # noqa: F401
        print(f"[OK] Ultralytics already installed: {ultralytics.__version__}")
    except ImportError:
        print("[INFO] Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("[OK] Ultralytics installed.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 terrain detector")
    parser.add_argument("--data",    required=True,           help="Path to dataset.yaml")
    parser.add_argument("--model",   default="yolov8n.pt",    help="YOLO base model weights")
    parser.add_argument("--epochs",  type=int,   default=100, help="Training epochs")
    parser.add_argument("--imgsz",   type=int,   default=640, help="Input image size (px)")
    parser.add_argument("--batch",   type=int,   default=16,  help="Batch size (-1 = auto)")
    parser.add_argument("--name",    default="terrain_detector", help="Run name (saved under runs/detect/)")
    parser.add_argument("--device",  default=None,            help="Device: cpu | 0 | 0,1 (default: auto)")
    parser.add_argument("--patience",type=int,   default=20,  help="Early stopping patience (epochs)")
    parser.add_argument("--workers", type=int,   default=4,   help="Dataloader workers")
    parser.add_argument("--resume",  action="store_true",     help="Resume last interrupted run")
    args = parser.parse_args()

    ensure_ultralytics()

    from ultralytics import YOLO  # import after ensure

    #  Load model 
    if args.resume:
        # Find the most recent last.pt
        from pathlib import Path
        candidates = sorted(Path("runs/detect").glob(f"{args.name}*/weights/last.pt"))
        if not candidates:
            sys.exit("No previous run found to resume. Remove --resume to start fresh.")
        weights = str(candidates[-1])
        print(f"[RESUME] Resuming from {weights}")
    else:
        weights = args.model

    model = YOLO(weights)

    #  Training hyperparameters
    train_kwargs = dict(
        data        = args.data,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        name        = args.name,
        patience    = args.patience,
        workers     = args.workers,
        # Augmentation — useful for small terrain datasets
        augment     = True,
        degrees     = 10.0,        # rotation ±10°
        flipud      = 0.5,         # vertical flip (terrain can look similar upside-down)
        fliplr      = 0.5,
        scale       = 0.4,         # scale jitter ±40%
        mosaic      = 1.0,         # mosaic augmentation
        hsv_h       = 0.015,
        hsv_s       = 0.5,
        hsv_v       = 0.3,
        # Logging
        plots       = True,
        save        = True,
        save_period = 10,          # checkpoint every N epochs
        verbose     = True,
    )

    if args.device is not None:
        # Parse "0,1" as a list for multi-GPU
        device = [int(d) for d in args.device.split(",")] if "," in args.device else args.device
        train_kwargs["device"] = device

    if args.resume:
        train_kwargs["resume"] = True

    #  Train ─
    print("\n Starting training ")
    print(f"  Model   : {weights}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print("─\n")

    results = model.train(**train_kwargs)

    #  Validate on test split 
    print("\n Running test-set validation ")
    best_weights = f"runs/detect/{args.name}/weights/best.pt"
    best_model   = YOLO(best_weights)
    metrics = best_model.val(data=args.data, split="test", imgsz=args.imgsz)

    print("\n Results ")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Best weights : {best_weights}")
    print("─")


if __name__ == "__main__":
    main()