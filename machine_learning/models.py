from django.db import models
from django.conf import settings
import ast
import json
# Create your models here.


class ImageRecognition(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    image = models.ImageField(upload_to='images/', blank=True, null=True)
    image_analysis_data = models.JSONField(null=True, blank=True)
    dominant_emotion = models.CharField(max_length=255, null=True, blank=True)

    def images_list(self):
        return ast.literal_eval(self.dominant_emotion)

class VideoRecognition(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    analysis_score = models.FloatField(default=0.0)
    word_per_minute = models.FloatField(null=True, blank=True)
    language_analysis = models.JSONField(null=True, blank=True,default=None)
    voice_modulation_analysis = models.JSONField(null=True, blank=True, default=None)
    energy_level_analysis = models.CharField(max_length=255, null=True, blank=True)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    filler_words_used = models.TextField(null=True, blank=True)
    frequently_used_word = models.TextField(null=True, blank=True)
    voice_emotion = models.TextField(null=True, blank=True)
    confidence = models.CharField(max_length=100,null=True, blank=True)
    eye_bling = models.CharField(max_length=100,null=True, blank=True)
    hand_movement = models.CharField(max_length=100,null=True, blank=True)
    eye_contact = models.CharField(max_length=100,null=True, blank=True)
    thanks_gesture = models.CharField(max_length=100,null=True, blank=True)
    greeting = models.CharField(max_length=100,null=True, blank=True)
    greeting_gesture = models.CharField(max_length=100,null=True, blank=True)
    voice_tone = models.CharField(max_length=100,null=True, blank=True)
    voice_pauses = models.CharField(max_length=100,null=True, blank=True)
    appropriate_facial = models.CharField(max_length=100,null=True, blank=True)
    body_posture = models.CharField(max_length=100,null=True, blank=True)

    def __str__(self):
        return self.name
    
    def filler_words(self):
        if self.filler_words_used:
            return json.loads(self.filler_words_used)
        else:
            return []

    def frequently_word(self):
        if self.frequently_used_word:
            return json.loads(self.frequently_used_word)
        else:
            return []
        
    def get_voice_emotion(self):
        if self.voice_emotion:
            return json.loads(self.voice_emotion)
        else:
            return []

class AnalyzePDF(models.Model):
    pdf_name = models.CharField(max_length=255, null=True, blank=True)
    pdf_file = models.FileField(upload_to='pdf/', null=True, blank=True)
    pdf_text = models.TextField()

    def __str__(self):
        return self.pdf_name
    

    
