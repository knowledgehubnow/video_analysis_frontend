from rest_framework import serializers
from .models import VideoRecognition

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = ('video_file',)

class VideoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"

