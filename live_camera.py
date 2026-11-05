"""Live webcam classification for glass, plastic, and aluminium bottles."""
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model_terbaik.keras"
CLASS_MAP_PATH = ROOT / "class_names.json"
CAMERA_SCAN_LIMIT = 6
CONFIDENCE_THRESHOLD = 0.60
PREDICT_EVERY_N_FRAMES = 5
IMAGE_SIZE = (224, 224)


def open_first_available_camera():
    for index in range(CAMERA_SCAN_LIMIT):
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            continue
        ok, frame = camera.read()
        if ok and frame is not None:
            print(f"Webcam detected at camera index {index}.")
            return camera, index
        camera.release()
    raise RuntimeError(
        "No working camera found. Connect/enable your webcam and close other apps using it."
    )


def switch_camera(current_index):
    for offset in range(1, CAMERA_SCAN_LIMIT + 1):
        index = (current_index + offset) % CAMERA_SCAN_LIMIT
        candidate = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if candidate.isOpened():
            ok, frame = candidate.read()
            if ok and frame is not None:
                return candidate, index
        candidate.release()
    return None, current_index


def predict_frame(model, frame, class_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    batch = preprocess_input(batch)
    scores = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(scores))
    return class_names[index], float(scores[index])


def main():
    if not MODEL_PATH.is_file() or not CLASS_MAP_PATH.is_file():
        raise FileNotFoundError(
            "Three-class model files are missing. Run "
            "python machine_model/machine_model/train.py first."
        )
    class_names = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    if class_names != ["aluminium", "glass", "plastic"]:
        raise ValueError(f"Unexpected class order in class_names.json: {class_names}")

    print("Loading trained model...")
    model = load_model(MODEL_PATH)
    camera, camera_index = open_first_available_camera()
    last_label, last_confidence, frame_count = "Waiting for frame...", 0.0, 0
    window_name = "Bottle Material Classifier - Q: quit | C: change camera"

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame could not be read.")
                break

            frame_count += 1
            if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                try:
                    last_label, last_confidence = predict_frame(model, frame, class_names)
                except Exception as error:
                    last_label, last_confidence = "Prediction error", 0.0
                    print(f"Prediction error: {error}")

            if last_confidence < CONFIDENCE_THRESHOLD:
                shown_label, text_color = "Uncertain / not recognized", (0, 165, 255)
            else:
                shown_label, text_color = last_label.capitalize(), (0, 220, 0)

            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 92), (25, 25, 25), -1)
            cv2.putText(frame, f"Material: {shown_label}", (24, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Confidence: {last_confidence:.1%}", (24, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Camera {camera_index} | Q: quit | C: switch camera",
                        (16, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                new_camera, new_index = switch_camera(camera_index)
                if new_camera is not None:
                    camera.release()
                    camera, camera_index = new_camera, new_index
                    last_label, last_confidence, frame_count = "Waiting for frame...", 0.0, 0
                    print(f"Switched to camera index {camera_index}.")
                else:
                    print("No other working camera was found.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        input("Press Enter to close...")
