from django.contrib import admin
from django.urls import path,include
from . import views
from .views import *

urlpatterns = [
    path('',views.scan_face,name = "scan_face"),
    path('analyzed/video/view/<int:video_id>/',views.analized_video_detail,name = "analized_video_detail"),
    path('analized/video/list/',views.analized_video_list,name = "analized_video_list"),
    path('video_detail/<int:video_id>/',views.video_detail,name = "video_detail"),
]
