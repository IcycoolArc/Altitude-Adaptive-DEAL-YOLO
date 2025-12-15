import os
from pathlib import Path
import random
import shutil
from typing import List, Tuple

import cv2
import numpy as np
import albumentations as A


# ============================================================
# CONFIGURATION
# ============================================================

# ---- PATHS -------------------------------------------------
# Root of your WAID dataset
DATA_ROOT = Path(r"C:\Users\Arc\Experiment-YOLO\datasets\WAID\train")

# Input image/label folders (standard YOLO layout)
IMG_DIR = DATA_ROOT / "images"
LABEL_DIR = DATA_ROOT / "labels"

# Option 2: separate output for augmented data
USE_SEPARATE_OUTPUT_DIRS = False  # True = put augmented into separate folders

if USE_SEPARATE_OUTPUT_DIRS:
    OUT_IMG_DIR = DATA_ROOT / "aug_images"
    OUT_LABEL_DIR = DATA_ROOT / "aug_labels"
else:
    # Write augmented images into the same folders as originals
    OUT_IMG_DIR = IMG_DIR
    OUT_LABEL_DIR = LABEL_DIR

# Make sure output dirs exist
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- AUGMENTATION SETTINGS ---------------------------------
# How many augmented variants to create per original image
AUGS_PER_IMAGE = 2

# Suffix to add to augmented image/label filenames
AUG_SUFFIX = "_alt"

# Option 1: avoid re-augmenting the same image multiple times
# We maintain a log file of already processed image stems
SKIP_ALREADY_AUGMENTED = True
AUG_LOG_PATH = DATA_ROOT / "altitude_augmented_stems.txt"

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# AUGMENTATION PIPELINE
# (Customize as you like)
# ============================================================

transform = A.Compose(
    [
        # Geometry / "altitude-ish" changes
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=5,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        # Some light photometric changes
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5,
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",          # assumes YOLO format: class x_center y_center w h (normalized)
        label_fields=["class_labels"],
        min_visibility=0.0001,
        check_each_transform=False,
    ),
)


# ============================================================
# UTILS
# ============================================================

def load_aug_log(log_path: Path) -> set:
    """Load set of image stems already augmented."""
    if not log_path.exists():
        return set()
    with log_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return set(lines)


def append_to_aug_log(log_path: Path, stem: str) -> None:
    """Append a single image stem to the augmented log."""
    with log_path.open("a", encoding="utf-8") as f:
        f.write(stem + "\n")


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO labels: each line -> class x_center y_center width height (normalized).
    Returns list of tuples: (cls, x, y, w, h)
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                # Skip malformed lines
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            boxes.append((cls, x, y, w, h))
    return boxes


def write_yolo_labels(label_path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    """Write YOLO labels to file."""
    with label_path.open("w", encoding="utf-8") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def valid_yolo_box(x: float, y: float, w: float, h: float) -> bool:
    """
    Check if a YOLO box is valid in normalized coords (0-1).
    We also enforce min positive width/height to prevent degenerate boxes.
    """
    eps = 1e-6
    if w <= eps or h <= eps:
        return False
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False
    # corners
    x_min = x - w / 2.0
    y_min = y - h / 2.0
    x_max = x + w / 2.0
    y_max = y + h / 2.0
    if x_min < 0.0 or y_min < 0.0 or x_max > 1.0 or y_max > 1.0:
        # Allow slightly out of bounds but clamp?
        # Here we choose to reject; you can clamp if preferred.
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True


def filter_valid_boxes(
    boxes: List[Tuple[int, float, float, float, float]]
) -> List[Tuple[int, float, float, float, float]]:
    """Keep only boxes that pass validity check."""
    return [b for b in boxes if valid_yolo_box(*b[1:])]


def get_image_paths(img_dir: Path) -> List[Path]:
    """Get all image files (jpg/png/jpeg) in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    for p in img_dir.iterdir():
        if p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def find_label_for_image(label_dir: Path, image_path: Path) -> Path:
    """Return expected label path for a given image."""
    return label_dir / f"{image_path.stem}.txt"


def make_augmented_name(orig_path: Path, aug_idx: int) -> Path:
    """Return new filename with AUG_SUFFIX and index before extension."""
    return orig_path.with_name(f"{orig_path.stem}{AUG_SUFFIX}{aug_idx}{orig_path.suffix}")


# ============================================================
# MAIN AUGMENTATION LOGIC
# ============================================================

def process_image(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    processed_stems: set,
) -> bool:
    """
    Process a single (image, label) pair.
    Returns True if augmentation was performed, False otherwise.
    """

    stem = image_path.stem

    # Option 1: skip if we've already augmented this stem (from a previous run)
    if SKIP_ALREADY_AUGMENTED and stem in processed_stems:
        print(f"[SKIP] {stem} already augmented (in log).")
        return False

    if not label_path.exists():
        print(f"[WARN] No label file for {image_path.name}; skipping.")
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image {image_path}; skipping.")
        return False

    height, width = img.shape[:2]

    # Read and validate boxes
    boxes = read_yolo_labels(label_path)
    if not boxes:
        print(f"[WARN] No valid boxes in {label_path.name}; skipping.")
        return False

    boxes = filter_valid_boxes(boxes)
    if not boxes:
        print(f"[WARN] All boxes invalid in {label_path.name}; skipping.")
        return False

    # Separate class labels & bboxes for albumentations
    class_labels = [b[0] for b in boxes]
    yolo_boxes = [b[1:] for b in boxes]  # (x, y, w, h)

    # Apply multiple augmentations
    aug_done = False
    for i in range(1, AUGS_PER_IMAGE + 1):
        try:
            transformed = transform(
                image=img,
                bboxes=yolo_boxes,
                class_labels=class_labels,
            )
        except Exception as e:
            print(f"[ERROR] Transform failed for {image_path.name} (aug {i}): {e}")
            continue

        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        # Rebuild YOLO tuples and filter again (post-transform)
        aug_box_tuples = []
        for cls, (x, y, w, h) in zip(aug_labels, aug_bboxes):
            if valid_yolo_box(x, y, w, h):
                aug_box_tuples.append((cls, float(x), float(y), float(w), float(h)))

        if not aug_box_tuples:
            print(f"[INFO] No valid boxes after transform for {image_path.name} (aug {i}); skipping this aug.")
            continue

        # Build output paths
        out_img_path = make_augmented_name(out_img_dir / image_path.name, i)
        out_label_path = make_augmented_name(out_label_dir / label_path.name, i)

        # Save augmented image and labels
        cv2.imwrite(str(out_img_path), aug_img)
        write_yolo_labels(out_label_path, aug_box_tuples)

        print(f"[OK] Saved {out_img_path.name} and {out_label_path.name}")
        aug_done = True

    # If at least one augmentation was successful, mark this stem as processed
    if aug_done and SKIP_ALREADY_AUGMENTED:
        append_to_aug_log(AUG_LOG_PATH, stem)
        processed_stems.add(stem)

    return aug_done


def main():
    # Sanity checks on paths
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")
    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    print(f"Using DATA_ROOT: {DATA_ROOT}")
    print(f"Images: {IMG_DIR}")
    print(f"Labels: {LABEL_DIR}")
    print(f"Output images: {OUT_IMG_DIR}")
    print(f"Output labels: {OUT_LABEL_DIR}")
    print(f"Skip already augmented: {SKIP_ALREADY_AUGMENTED}")
    print(f"Separate output dirs: {USE_SEPARATE_OUTPUT_DIRS}")

    # Load augment log
    processed_stems = load_aug_log(AUG_LOG_PATH) if SKIP_ALREADY_AUGMENTED else set()
    if SKIP_ALREADY_AUGMENTED:
        print(f"Loaded {len(processed_stems)} already-augmented stems from log.")

    # Get all images
    image_paths = get_image_paths(IMG_DIR)
    print(f"Found {len(image_paths)} images to consider.")

    total_aug = 0
    total_skipped = 0

    for img_path in image_paths:
        lbl_path = find_label_for_image(LABEL_DIR, img_path)
        if process_image(img_path, lbl_path, OUT_IMG_DIR, OUT_LABEL_DIR, processed_stems):
            total_aug += 1
        else:
            total_skipped += 1

    print("============================================")
    print(f"Augmentation complete.")
    print(f"Images augmented (at least one variant): {total_aug}")
    print(f"Images skipped (no valid aug or already processed): {total_skipped}")
    print("============================================")


if __name__ == "__main__":
    main()
