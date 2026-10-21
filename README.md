# Waste Image Classifier

A TensorFlow/Keras image classifier that predicts one of 11 waste-item classes:
Beer, Glass cup, HDPE, Mug, Other, PET, Plastic cup, Soda, Tin, Tumbler, and
Wine. The prediction script opens a file picker and prints the predicted class
and score.

## Project files

- `machine_model/machine_model/train.py` trains the classifier.
- `machine_model/machine_model/test_model.py` predicts the class of a selected image.
- `model_terbaik.h5` is the trained model used by the prediction script.
- `assets/` is the local training dataset and is intentionally excluded from
  this public Git repository. Do not redistribute images unless their licenses
  permit it.

## Requirements

- Python with TensorFlow available for your operating system and Python version.
- Dependencies listed in `requirements.txt`.
- Tkinter for the image picker (usually included with desktop Python on Windows).

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Predict an image

From the project root, run:

```bash
python machine_model/machine_model/test_model.py
```

Choose an image in the file picker. The script loads `model_terbaik.h5` from
the project root, regardless of the current working directory.

## Train the model

Place the training images under `assets/`, with one subdirectory per class.
For example:

```text
assets/
  beer/
  glass_cup/
  hdpe/
  mug/
  other/
  pet/
  plastic_cup/
  soda/
  tin/
  tumbler/
  wine/
```

Run training from the project root:

```bash
python machine_model/machine_model/train.py
```

The best validation-accuracy model is saved as `model_terbaik.h5` in the
project root. Keep a local copy of the dataset, or distribute it through a
separate dataset host only when you have the right to share it. Git LFS can
store large files in a GitHub repository, but has separate storage and
bandwidth limits.
