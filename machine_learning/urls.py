from django.contrib import admin
from django.urls import path,include
from . import views
from .views import *

urlpatterns = [
    path('',views.scan_face,name = "scan_face"),
    path('image/processing/',views.image_analysis,name = "image_analysis"),
    path('analyzed/video/view/<int:video_id>/',views.analized_video_detail,name = "analized_video_detail"),
    path('analized/video/list/',views.analized_video_list,name = "analized_video_list"),
    path('analized/image/list/',views.analized_image_list,name = "analized_image_list"),
    path('analyze/pdf/',views.analyze_pdf,name = "analyze_pdf"),
    path('analyzed/pdf/list/',views.analyzed_pdf_list,name = "analyzed_pdf_list"),
    path('analyzed/pdf/view/<int:pdf_id>/',views.analyzed_pdf_view,name = "analyzed_pdf_view"),
    path('upload/', VideoUploadView.as_view(), name='video_upload'),
    path('scan/live/face/',views.scan_live_face,name = "scan_live_face"),
    path('live/video/analysis/', LiveVideoAnalysisView.as_view(), name='live_video'),
    path('live/analysis/', LiveAnalysisView.as_view(), name='live_analysis'),
    path('image/analysis/', ImageAnalysisView.as_view(), name='image_analysi')
]