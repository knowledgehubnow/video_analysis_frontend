from django.contrib import admin
from .models import *
# Register your models here.

admin.site.register(ImageRecognition)
admin.site.register(VideoRecognition)
admin.site.register(AnalyzePDF)
admin.site.register(DetectedFrames)
admin.site.register(Posture)
admin.site.register(Frame)
