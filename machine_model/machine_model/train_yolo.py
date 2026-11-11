"""Train a four-class YOLO bottle-material detector from YOLO annotations."""
import random
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "assets" / "YOLO_imgs"
WORK_DIR = Path(__file__).resolve().parent
TRAIN_LIST_PATH = WORK_DIR / "yolo_train.txt"
VAL_LIST_PATH = WORK_DIR / "yolo_val.txt"
DATA_CONFIG_PATH = WORK_DIR / "bottle_dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"
CLASS_NAMES = ["aluminium", "glass", "hdpe", "plastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
VALIDATION_SPLIT = 0.2
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 32          # RTX 4060 8 GB VRAM – safe at 640 px with yolo11n/s
WORKERS = 8              # parallel data-loading threads
CACHE = "ram"            # pre-load dataset into RAM for faster epoch iteration; use "disk" if RAM < 8 GB


def valid_label_file(path: Path) -> bool:
    """Return True only for non-empty, valid YOLO label files."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return False

    for line in lines:
        values = line.split()
        if len(values) != 5:
            return False
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError:
            return False
        if class_id not in range(len(CLASS_NAMES)):
            return False
        if not all(0.0 <= coordinate <= 1.0 for coordinate in coordinates):
            return False
    return True


def prepare_dataset_lists() -> tuple[int, int, int]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"YOLO dataset folder not found: {DATA_DIR}")

    valid_images = []
    skipped = 0
    for image_path in sorted(DATA_DIR.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file() or not valid_label_file(label_path):
            skipped += 1
            continue
        valid_images.append(image_path.resolve().as_posix())

    if len(valid_images) < len(CLASS_NAMES) * 5:
        raise ValueError("Not enough valid labeled images for training.")

    random.Random(SEED).shuffle(valid_images)
    validation_count = max(1, round(len(valid_images) * VALIDATION_SPLIT))
    val_images = valid_images[:validation_count]
    train_images = valid_images[validation_count:]

    TRAIN_LIST_PATH.write_text("\n".join(train_images) + "\n", encoding="utf-8")
    VAL_LIST_PATH.write_text("\n".join(val_images) + "\n", encoding="utf-8")
    dataset_config = {
        "train": TRAIN_LIST_PATH.resolve().as_posix(),
        "val": VAL_LIST_PATH.resolve().as_posix(),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    DATA_CONFIG_PATH.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8"
    )
    return len(train_images), len(val_images), skipped


def main():
    train_count, val_count, skipped = prepare_dataset_lists()
    print(
        f"Prepared {train_count} training and {val_count} validation images "
        f"({skipped} invalid or unlabeled images skipped)."
    )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Ultralytics training device: {device}")

    # Use yolo11s (small) instead of nano – better accuracy, still fits 8 GB VRAM at batch=32
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA_CONFIG_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        cache=CACHE,            # pre-load dataset into RAM/disk for faster iterations
        amp=True,               # mixed-precision (FP16) – halves VRAM usage, 2× faster
        optimizer="AdamW",      # converges faster than SGD for small datasets
        lr0=0.001,              # initial learning rate
        cos_lr=True,            # cosine LR decay for smoother convergence
        patience=20,            # stop early if no improvement for 20 epochs
        project=str(RUNS_DIR),
        name="bottle_detector",
        exist_ok=True,
        verbose=True,
    )
    print(f"Best detector: {RUNS_DIR / 'bottle_detector' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
