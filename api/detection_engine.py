import os
import cv2
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image

from yolo.detector import model
from .models import Alert
from django.core.files.base import ContentFile


# ==============================
# BASE DIRECTORY
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# CONFIGURATION (Extreme Recall Mode)
# ==============================

GUN_THRESHOLD = 0.50
KNIFE_THRESHOLD = 0.35

MAX_CANDIDATES_PER_CLASS = 2
ALERT_COOLDOWN = 10
FRAME_SKIP = 15
INFERENCE_SIZE = 640

CLASSIFIER_THRESHOLD = 0.80

last_alert_time = 0


# ==============================
# LOAD CLASSIFIER MODEL
# ==============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3  # background, gun, knife (alphabetical order)

classifier_model = models.efficientnet_b0(weights=None)
classifier_model.classifier[1] = torch.nn.Linear(
    classifier_model.classifier[1].in_features,
    NUM_CLASSES
)

CLASSIFIER_PATH = os.path.join(BASE_DIR, "classifier", "best_classifier.pth")

classifier_model.load_state_dict(
    torch.load(CLASSIFIER_PATH, map_location=DEVICE)
)

classifier_model.to(DEVICE)
classifier_model.eval()

# IMPORTANT: Match ImageFolder alphabetical order
CLASS_NAMES = ["background", "gun", "knife"]

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# STAGE 2 - CLASSIFIER
# ==============================

def weapon_classifier(cropped_img, expected_label):

    if cropped_img is None or cropped_img.size == 0:
        return False

    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cropped_img)

    input_tensor = CLASSIFIER_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted.item()]
    confidence = confidence.item()

    print(f"[CLASSIFIER] {predicted_label} ({round(confidence, 2)})")

    # Reject background
    if predicted_label == "background":
        return False

    # Accept only if matches YOLO label and passes threshold
    if predicted_label == expected_label and confidence >= CLASSIFIER_THRESHOLD:
        return True

    return False


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

        if not ret:
            print("[WARNING] Frame read failed. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(source)
            continue

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            continue

        results = model(frame, imgsz=INFERENCE_SIZE, verbose=False)

        gun_candidates = []
        knife_candidates = []

        # STAGE 1 - YOLO
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

        gun_candidates.sort(reverse=True)
        knife_candidates.sort(reverse=True)

        gun_candidates = gun_candidates[:MAX_CANDIDATES_PER_CLASS]
        knife_candidates = knife_candidates[:MAX_CANDIDATES_PER_CLASS]

        current_time = time.time()

        # VALIDATE GUNS
        for conf, x1, y1, x2, y2 in gun_candidates:

            print(f"[CANDIDATE] GUN ({round(conf,2)})")

            h, w, _ = frame.shape
            pad = 0.1

            dx = int((x2 - x1) * pad)
            dy = int((y2 - y1) * pad)

            x1_p = max(0, x1 - dx)
            y1_p = max(0, y1 - dy)
            x2_p = min(w, x2 + dx)
            y2_p = min(h, y2 + dy)

            cropped = frame[y1_p:y2_p, x1_p:x2_p]

            if weapon_classifier(cropped, "gun"):
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    print(f"[CONFIRMED ALERT] GUN verified ({round(conf,2)})")
                    save_alert(frame, "gun", conf)
                    last_alert_time = current_time

        # VALIDATE KNIVES
        for conf, x1, y1, x2, y2 in knife_candidates:

            print(f"[CANDIDATE] KNIFE ({round(conf,2)})")

            h, w, _ = frame.shape
            pad = 0.1

            dx = int((x2 - x1) * pad)
            dy = int((y2 - y1) * pad)

            x1_p = max(0, x1 - dx)
            y1_p = max(0, y1 - dy)
            x2_p = min(w, x2 + dx)
            y2_p = min(h, y2 + dy)

            cropped = frame[y1_p:y2_p, x1_p:x2_p]

            if weapon_classifier(cropped, "knife"):
                if current_time - last_alert_time > ALERT_COOLDOWN:
                    print(f"[CONFIRMED ALERT] KNIFE verified ({round(conf,2)})")
                    save_alert(frame, "knife", conf)
                    last_alert_time = current_time

        time.sleep(0.05)