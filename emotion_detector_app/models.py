from django.db import models
from django.utils import timezone

# Create your models here.
class EmotionRecord(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    image = models.ImageField(upload_to='uploads/')
    username = models.CharField(max_length=30, default='Anonymous')
    emotion = models.CharField(max_length=20)
    confidence = models.FloatField()

    def __str__(self):
        return f"{self.emotion} ({self.confidence:.2f}) at {self.timestamp}"