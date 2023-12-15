import requests

url = "https://ai.eduagentapp.com/live/video/analysis/"

payload = {}
files=[
  ('video_file',('/home/manish/Downloads/c9.mp4',open('/home/manish/Downloads/c9.mp4','rb'),'application/octet-stream'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)
print(response)
print(response.text)
