import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from yolo.detector import detect_weapons
from yolo.detector import detect_video
from .models import Alert

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

    # Run detection
    detections = detect_weapons(image_path)

    return JsonResponse({
        "detections": detections
    })


@csrf_exempt
def detect_video_api(request):
    if request.method != 'POST':
        return JsonResponse({"error": "POST request required"}, status=400)

    if 'video' not in request.FILES:
        return JsonResponse({"error": "No video provided"}, status=400)

    video_file = request.FILES['video']

    video_path = os.path.join(settings.MEDIA_ROOT, "upload_" + video_file.name)
    with open(video_path, 'wb+') as f:
        for chunk in video_file.chunks():
            f.write(chunk)

    detections = detect_video(video_path)

    return JsonResponse({
        "detections": detections
    })



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
