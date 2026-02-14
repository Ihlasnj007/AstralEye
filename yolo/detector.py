# yolo/detector.py
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "yolo", "best.pt")

# Load model once when Django starts
model = YOLO(MODEL_PATH)

def detect_weapons(image_path):
    """
    Runs YOLO inference on an image
    Returns list of detected weapons
    """
    results = model(image_path)

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