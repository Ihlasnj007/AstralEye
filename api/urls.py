from django.urls import path
from .views import detect_api, detect_video_api, get_alerts

urlpatterns = [
    path('detect/', detect_api, name="detect_endpoint"),
    path('detect-video/', detect_video_api, name="video_endpoint"),
    path('alerts/', get_alerts, name="alerts_endpoint"),
]
