import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml   # pip install pyyaml

CLASSES = ["hole", "mound"]   # class index 0, 1  must match mask_to_yolo.py


def split_files(files, train_frac, val_frac, seed):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = files[:n_train]
    val   = files[n_train:n_train + n_val]
    test  = files[n_train + n_val:]
    return train, val, test


def copy_or_link(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits")
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--label_dir",  required=True)
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.20)
    parser.add_argument("--test",  type=float, default=0.10)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--copy",  action="store_true",
                        help="Copy files instead of symlinking (needed on Windows or cross-device)")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        sys.exit("--train + --val + --test must sum to 1.0")

    image_dir  = Path(args.image_dir)
    label_dir  = Path(args.label_dir)
    output_dir = Path(args.output_dir)

    # Gather paired image+label files
    image_files = sorted(f for f in image_dir.glob("*")
                         if f.suffix.lower() in {".png", ".jpg", ".jpeg"})

    paired = []
    missing_labels = []
    for img in image_files:
        lbl = label_dir / img.with_suffix(".txt").name
        if lbl.exists():
            paired.append((img, lbl))
        else:
            missing_labels.append(img.name)

    if missing_labels:
        print(f"[WARN] {len(missing_labels)} images have no label file and will be skipped:")
        for m in missing_labels[:10]:
            print(f"       {m}")
        if len(missing_labels) > 10:
            print(f"       ... and {len(missing_labels)-10} more")

    if not paired:
        sys.exit("No paired image+label files found.")

    train_pairs, val_pairs, test_pairs = split_files(
        paired, args.train, args.val, args.seed)

    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split, pairs in splits.items():
        for img_src, lbl_src in pairs:
            img_dst = output_dir / "images" / split / img_src.name
            lbl_dst = output_dir / "labels" / split / lbl_src.name
            copy_or_link(img_src, img_dst, args.copy)
            copy_or_link(lbl_src, lbl_dst, args.copy)

    yaml_path = output_dir / "dataset.yaml"
    dataset_cfg = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)

    print("\n Dataset prepared ─")
    print(f"  Train : {len(train_pairs)} samples")
    print(f"  Val   : {len(val_pairs)}   samples")
    print(f"  Test  : {len(test_pairs)}  samples")
    print(f"  YAML  : {yaml_path}")
    print("─")
    print("\nNext step:  python train_yolo.py --data", yaml_path)


if __name__ == "__main__":
    main()