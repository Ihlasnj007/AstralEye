from django.urls import path
from .views import detect_api

urlpatterns = [
    path('detect/', detect_api),
]
