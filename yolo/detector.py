# yolo/detector.py

import os
import torch
from ultralytics import YOLO


# ==============================
# MODEL PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "yolo", "best.pt")


# ==============================
# DEVICE SELECTION
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading YOLO model on device: {device}")


# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)
model.to(device)

print("[INFO] YOLO model loaded successfully.")