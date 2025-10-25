from django.db import models

class ChatMessage(models.Model):
    session_key = models.CharField(max_length=40)
    sender = models.CharField(max_length=10)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender}: {self.message[:30]}"
