"""Train a four-class bottle-material classifier (aluminium, glass, plastic, hdpe)."""
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "assets"
MODEL_PATH = PROJECT_ROOT / "model_terbaik.keras"
CLASS_MAP_PATH = PROJECT_ROOT / "class_names.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
EXPECTED_CLASSES = ["aluminium", "glass", "plastic", "hdpe"]
SEED = 42


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}. Create assets/aluminium, assets/glass, assets/plastic, and assets/hdpe first.",
        )

    found_classes = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}
    expected_classes = set(EXPECTED_CLASSES)
    if found_classes != expected_classes:
        raise ValueError(
            "Dataset must contain exactly "
            f"{sorted(expected_classes)}; found {sorted(found_classes)}"
        )

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts = {
        name: sum(1 for p in (DATA_DIR / name).rglob("*")
                  if p.is_file() and p.suffix.lower() in extensions)
        for name in EXPECTED_CLASSES
    }
    print("Dataset image counts:", counts)
    insufficient = {k: v for k, v in counts.items() if v < 5}
    if insufficient:
        raise ValueError(f"At least 5 images per class are required: {insufficient}")

    CLASS_MAP_PATH.write_text(json.dumps(EXPECTED_CLASSES, indent=2), encoding="utf-8")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )
    train_data = train_datagen.flow_from_directory(
        str(DATA_DIR), classes=EXPECTED_CLASSES, target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="training", shuffle=True, seed=SEED,
    )
    val_data = val_datagen.flow_from_directory(
        str(DATA_DIR), classes=EXPECTED_CLASSES, target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical",
        subset="validation", shuffle=False, seed=SEED,
    )

    print("Class index mapping:", train_data.class_indices)
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(len(EXPECTED_CLASSES), activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3,
            min_lr=1e-6, verbose=1,
        ),
    ]
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
    print(f"Training complete. Best model: {MODEL_PATH}")
    print(f"Class mapping: {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()
