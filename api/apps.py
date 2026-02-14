from django.apps import AppConfig
import threading
import os

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        if os.environ.get('RUN_MAIN') != 'true':
            return

        from .detection_engine import start_detection

        ip_camera_url = "http://10.93.126.70:8080/video"

        threading.Thread(
            target=start_detection,
            args=(ip_camera_url,),
            daemon=True
        ).start()
