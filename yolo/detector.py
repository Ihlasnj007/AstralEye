# yolo/detector.py

import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolo", "best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Loading model on device: {device}")

model = YOLO(MODEL_PATH)

# Let Ultralytics handle precision internally
model.to(device)


def detect_weapons(image_path, imgsz=640):
    results = model(image_path, imgsz=imgsz, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    return detections


def detect_video(video_path, imgsz=640):
    results = model(video_path, imgsz=imgsz, verbose=False)

    detection_summary = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)

            detection_summary.append({
                "label": model.names[cls_id],
                "confidence": round(conf, 3)
            })

    return detection_summary
