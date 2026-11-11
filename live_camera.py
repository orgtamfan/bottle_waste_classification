"""Live webcam detection for aluminium, glass, HDPE, and plastic bottles."""
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "bottle_detector" / "weights" / "best.pt"
CAMERA_SCAN_LIMIT = 6
CONFIDENCE_THRESHOLD = 0.40
PREDICT_EVERY_N_FRAMES = 3
DETECTOR_IMAGE_SIZE = 640
BOX_COLORS = [
    (0, 180, 255),
    (255, 180, 0),
    (180, 0, 255),
    (0, 220, 80),
]


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


def detect_objects(model, frame):
    result = model(frame, imgsz=DETECTOR_IMAGE_SIZE,
                   conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = result.names[class_id]
        detections.append((x1, y1, x2, y2, class_id, class_name, confidence))
    return detections


def draw_detections(frame, detections):
    for x1, y1, x2, y2, class_id, class_name, confidence in detections:
        color = BOX_COLORS[class_id % len(BOX_COLORS)]
        label = f"{class_name}: {confidence:.1%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = max(0, y1 - label_height - baseline - 8)
        cv2.rectangle(frame, (x1, label_top),
                      (x1 + label_width + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2,
                    cv2.LINE_AA)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "YOLO detector model is missing. Run "
            "python machine_model/machine_model/train_yolo.py first."
        )

    print("Loading trained YOLO detector...")
    model = YOLO(str(MODEL_PATH))
    camera, camera_index = open_first_available_camera()
    detections, frame_count = [], 0
    window_name = "Bottle Material Detector - Q: quit | C: change camera"

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame could not be read.")
                break

            frame_count += 1
            if frame_count % PREDICT_EVERY_N_FRAMES == 0:
                try:
                    detections = detect_objects(model, frame)
                except Exception as error:
                    detections = []
                    print(f"Prediction error: {error}")

            draw_detections(frame, detections)
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 48), (25, 25, 25), -1)
            cv2.putText(frame, f"Detected bottles: {len(detections)}", (24, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2,
                        cv2.LINE_AA)
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
                    detections, frame_count = [], 0
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
