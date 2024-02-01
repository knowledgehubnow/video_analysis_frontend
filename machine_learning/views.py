from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import requests

  

def scan_face(request):
    if request.method == "POST":
        video = request.FILES['video']  # get the uploaded file
        url = "http://115.241.73.227/video/analysis/"

        payload = {'user': '8299037804'}
        files = {'video_file': video} 
        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        print(response.text)
        return HttpResponse(response.text)
    return render(request,"upload.html")

def analized_video_detail(request):
    return render(request,"video_detail.html")

def analized_video_list(request):
    user = "8299037804"
    url = f"http://115.241.73.227/video/analysed/video/list/?user={8299037804}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    response_data = response.json()
    print(response_data)
    return render(request,"analize_video_list.html",{
        "all_data":response_data["data"],
    })


def video_detail(request,video_id):
    return render(request,"detail.html")

            