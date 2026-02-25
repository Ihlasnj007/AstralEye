import cv2
import time
import torch
from yolo.detector import model
from .models import Alert
from django.core.files.base import ContentFile


# ==============================
# CONFIGURATION (Extreme Recall Mode)
# ==============================

GUN_THRESHOLD = 0.50
KNIFE_THRESHOLD = 0.35

MAX_CANDIDATES_PER_CLASS = 2   # Prevent classifier overload
ALERT_COOLDOWN = 10
FRAME_SKIP = 15
INFERENCE_SIZE = 640

last_alert_time = 0


# ==============================
# STAGE 2 - CLASSIFIER (GPU READY PLACEHOLDER)
# ==============================

def weapon_classifier(cropped_img, label):
    """
    Replace this with real CNN classifier.
    Runs on GPU later.
    """

    if cropped_img is None:
        return False

    # Placeholder: Always accept
    return True


# ==============================
# SAVE ALERT
# ==============================

def save_alert(frame, label, confidence):
    _, buffer = cv2.imencode('.jpg', frame)

    image_file = ContentFile(
        buffer.tobytes(),
        name=f"alert_{int(time.time())}.jpg"
    )

    Alert.objects.create(
        label=label,
        confidence=round(confidence, 3),
        snapshot=image_file
    )


# ==============================
# MAIN DETECTION ENGINE
# ==============================

def start_detection(source=None):
    global last_alert_time

    print("[INFO] High-Recall Two-Stage Engine Started")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Unable to open camera stream")
        return

    print("[INFO] Camera connected successfully")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        # Auto reconnect
        if not ret:
            print("[WARNING] Frame read failed. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(source)
            continue

        frame_count += 1

        # Performance control
        if frame_count % FRAME_SKIP != 0:
            continue

        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)

        gun_candidates = []
        knife_candidates = []

        # ==============================
        # STAGE 1 - HIGH RECALL DETECTION
        # ==============================

        for result in results:
            for box in result.boxes:

                conf = float(box.conf)
                cls_id = int(box.cls)
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "gun" and conf >= GUN_THRESHOLD:
                    gun_candidates.append((conf, x1, y1, x2, y2))

                elif label == "knife" and conf >= KNIFE_THRESHOLD:
                    knife_candidates.append((conf, x1, y1, x2, y2))

        # Sort by confidence (highest first)
        gun_candidates.sort(reverse=True)
        knife_candidates.sort(reverse=True)

        # Limit candidates per frame
        gun_candidates = gun_candidates[:MAX_CANDIDATES_PER_CLASS]
        knife_candidates = knife_candidates[:MAX_CANDIDATES_PER_CLASS]

        # ==============================
        # STAGE 2 - CLASSIFIER VALIDATION
        # ==============================

        current_time = time.time()

        for conf, x1, y1, x2, y2 in gun_candidates:

            print(f"[CANDIDATE] GUN ({round(conf,2)})")

            cropped = frame[y1:y2, x1:x2]

            if weapon_classifier(cropped, "gun"):

                if current_time - last_alert_time > ALERT_COOLDOWN:

                    print(f"[CONFIRMED ALERT] GUN verified ({round(conf,2)})")

                    save_alert(frame, "gun", conf)
                    last_alert_time = current_time

        for conf, x1, y1, x2, y2 in knife_candidates:

            print(f"[CANDIDATE] KNIFE ({round(conf,2)})")

            cropped = frame[y1:y2, x1:x2]

            if weapon_classifier(cropped, "knife"):

                if current_time - last_alert_time > ALERT_COOLDOWN:

                    print(f"[CONFIRMED ALERT] KNIFE verified ({round(conf,2)})")

                    save_alert(frame, "knife", conf)
                    last_alert_time = current_time

        time.sleep(0.05)