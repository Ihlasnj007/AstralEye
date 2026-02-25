import os
import cv2
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from yolo.detector import model
from .models import Alert


# ==============================
# IMAGE DETECTION API
# ==============================

@csrf_exempt
def detect_api(request):

    if request.method != 'POST':
        return JsonResponse({"error": "POST request required"}, status=400)

    if 'image' not in request.FILES:
        return JsonResponse({"error": "No image provided"}, status=400)

    image_file = request.FILES['image']

    # Save image temporarily
    image_path = os.path.join(settings.MEDIA_ROOT, "upload_" + image_file.name)
    with open(image_path, 'wb+') as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Read image using OpenCV
    frame = cv2.imread(image_path)

    # Run YOLO inference
    results = model(frame, imgsz=640, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)

            detections.append({
                "label": model.names[cls_id],
                "confidence": round(conf, 3)
            })

    return JsonResponse({"detections": detections})


# ==============================
# GET ALERTS (REAL-TIME SYSTEM)
# ==============================

def get_alerts(request):

    alerts = Alert.objects.filter(is_viewed=False).order_by('-created_at')

    data = []

    for alert in alerts:
        data.append({
            "id": alert.id,
            "label": alert.label,
            "confidence": alert.confidence,
            "snapshot": request.build_absolute_uri(alert.snapshot.url),
            "created_at": alert.created_at
        })

        alert.is_viewed = True
        alert.save()

    return JsonResponse({"alerts": data})