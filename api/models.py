from django.db import models
from django.db import models

# Create your models here.


class Alert(models.Model):
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    snapshot = models.ImageField(upload_to="alerts/")
    created_at = models.DateTimeField(auto_now_add=True)
    is_viewed = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.label} - {self.confidence}"
