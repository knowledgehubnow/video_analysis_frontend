from rest_framework import serializers
from .models import VideoRecognition,ImageRecognition

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = ('video_file',)

class VideoDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"

class VideoDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoRecognition
        fields = "__all__"

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = ('image',)

class ImageDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = "__all__"

class ImageDataListSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageRecognition
        fields = "__all__"


