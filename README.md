# Bottle Material Classification

A Windows desktop application that uses a webcam and a TensorFlow/Keras
MobileNetV2 model to classify a bottle image into exactly four material
classes:

- **aluminium** — aluminium containers / cans
- **glass** — glass containers
- **plastic** — plastic containers (excluding HDPE bottles)
- **hdpe** — HDPE containers

The app shows a live camera preview, predicts periodically, displays a
confidence score, and can switch between detected cameras.

> This project is an image **classifier**, not an object detector. It classifies
> the whole camera frame; it does not draw bounding boxes or independently
> classify several bottles in one frame. Keep one bottle prominent in the view
> for the most meaningful prediction.

## Project files

- `live_camera.py` — live webcam classification app.
- `machine_model/machine_model/train.py` — trains the four-class model.
- `machine_model/machine_model/test_model.py` — backward-compatible launcher
  for the live camera app.
- `class_names.json` — exact output label order.
- `model_terbaik.keras` — generated model file; not committed to Git.
- `assets/` — local training dataset; intentionally excluded from Git.

## 1. Prepare the dataset

Create this structure in the repository root:

```text
assets/
  aluminium/
  glass/
  hdpe/
  plastic/
```

Put only images that belong to the matching class into each folder. The
training script uses the folder name as the image label and ignores any YOLO
`.txt` annotation files that may sit beside the images. Inspect each image and
assign it to its actual material class; filenames alone may not be reliable.

Remove unrelated objects such as mugs, cups, and tumblers unless the research
scope explicitly includes them as bottles. Aim for a balanced dataset with
varied lighting, backgrounds, camera angles, bottle colors, and shapes. Keep
separate real-world test images out of the training dataset.

The training script checks for at least 5 images per class to make a split.
This is only a technical minimum, not enough data for a reliable model.
More varied, correctly labeled examples are strongly recommended.

**Class definition:** decide whether the aluminium class means aluminium
bottles only or also beverage cans. If the research topic is specifically
bottles, do not silently include cans; define and apply the scope consistently.

## 2. Install dependencies (Windows)

Open PowerShell or Command Prompt in the repository root. Create and activate
a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow installation depends on the installed Python version and hardware.
If installation fails, use a Python version supported by the TensorFlow release
available for your Windows setup.

## 3. Train the four-class model

From the repository root:

```powershell
python machine_model/machine_model/train.py
```

The script checks that the dataset contains exactly the four expected
folders, uses an 80/20 training/validation split, and saves:

- `model_terbaik.keras` — best model selected by validation accuracy.
- `class_names.json` — exact output order used by the model.

The model file is generated locally and is not included in this repository.
Do not run the webcam app until training has completed successfully.

## 4. Run live webcam classification

From the repository root:

```powershell
python live_camera.py
```

Controls:

- **Q** — quit the application.
- **C** — switch to another detected camera.

The app scans camera indexes 0–5 and opens the first working camera. It runs
inference every few frames to keep the preview responsive. Predictions below
60% confidence are displayed as **Uncertain / not recognized**. This threshold
is a starting point and should be calibrated using a separate validation/test
set; it is not a guarantee that an uncertain prediction is truly an unknown
object.

You can also launch it through the old script path:

```powershell
python machine_model/machine_model/test_model.py
```

## Evaluation and limitations

- The model predicts one label for the entire frame. Keep one bottle centered
  and clearly visible.
- It does not detect bottle boundaries, count objects, or handle multiple
  bottles independently.
- Confidence is not the same as accuracy. Evaluate on held-out images and
  report a confusion matrix, precision, recall, and F1-score.
- Avoid near-duplicate photos of the same physical bottle in both training and
  validation/test sets; this can make evaluation look better than real-world
  performance.
