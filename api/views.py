import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from yolo.detector import detect_weapons


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
