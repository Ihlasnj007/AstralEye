import cv2
import time
import os
from django.conf import settings
from yolo.detector import model
from .models import Alert
from django.core.files.base import ContentFile

# ==============================
# CONFIGURATION
# ==============================

CONF_THRESHOLD = 0.5          # Minimum confidence to trigger alert
ALERT_COOLDOWN = 10           # Seconds between alerts
FRAME_SKIP = 15               # Process 1 frame every 15 frames
INFERENCE_SIZE = 640          # Increase detection resolution

last_alert_time = 0


# ==============================
# AUTO CAMERA DETECTION
# ==============================

def get_available_camera():
    """
    Tries camera indexes 0-4 and returns the first working one.
    """
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Camera found at index {i}")
            cap.release()
            return i
    return None


# ==============================
# MAIN DETECTION ENGINE
# ==============================

def start_detection(source=None):
    global last_alert_time

    print("[INFO] Starting Detection Engine...")

    # If no source provided → auto detect camera
    if source is None:
        source = get_available_camera()

        if source is None:
            print("[ERROR] No camera found. Detection engine stopped.")
            return

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Camera could not be opened.")
        return

    print("[INFO] Camera opened successfully.")
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame_count += 1

        # Frame skipping
        if frame_count % FRAME_SKIP != 0:
            continue

        # Run YOLO inference
        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)

        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                cls_id = int(box.cls)

                print(f"[DEBUG] Detected: {model.names[cls_id]} | Confidence: {conf}")

                if conf > CONF_THRESHOLD:
                    current_time = time.time()

                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        label = model.names[cls_id]

                        print(f"[ALERT] {label} detected with {conf}")

                        # Save snapshot
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_file = ContentFile(
                            buffer.tobytes(),
                            name=f"alert_{int(current_time)}.jpg"
                        )

                        Alert.objects.create(
                            label=label,
                            confidence=round(conf, 3),
                            snapshot=image_file
                        )

                        last_alert_time = current_time

        time.sleep(0.05)

    cap.release()
    print("[INFO] Detection engine stopped.")
