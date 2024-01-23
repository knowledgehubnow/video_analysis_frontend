from django.shortcuts import render, redirect,HttpResponse
from django.http import JsonResponse
import requests

  

def scan_face(request):
    if request.method == "POST":
        image = request.FILES['video']  # get the uploaded file
        url = "http://115.241.73.227/upload/"

        payload = {'user': '8299037804'}
        files=[
        ('video_file',(image,open(image,'rb'),'application/octet-stream'))
        ]
        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        print(response.text)
        print(image)
        return HttpResponse(response.text)
    return render(request,"upload.html")

def analized_video_detail(request):
    return render(request,"video_detail.html")

def analized_video_list(request):
    user = 8299037804
    url = f"http://115.241.73.227/analysed/video/list/?user={user}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)
    return render(request,"analize_video_list.html",{
        "all_data":response.text
    })


def video_detail(request,video_id):
    return render(request,"detail.html")

            