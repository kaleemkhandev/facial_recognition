from fastapi import FastAPI, UploadFile, File,Query,Body
import cv2
import numpy as np
import urllib
import urllib.parse
from typing import List,Dict
import uvicorn
from main import run_on_url_list,get_facial_encodings,Gestures_Check

app = FastAPI()

# @app.get("/process_video")
# async def process_video(vid_file_path):
#     # Create a VideoCapture object
#     print("hello")
#     video_capture = cv2.VideoCapture(vid_file_path)
#     # Set the video source as the retrieved video file
#     name = run(video_capture)
#     return {"Names":name}

@app.post("/signin")
async def signin(data:Dict[str,List[str]]=Body(...)):
    # Create a VideoCapture object
    try:
        print("hello")
        identifier = list(data.keys())[0]
        url_list = data[identifier]
        # img_url = "https://prod-images.tcm.com/Master-Profile-Images/LeonardoDiCaprio.jpg"
        
        # Set the video source as the retrieved video file
        response = run_on_url_list(url_list,identifier)
        return response 
    except Exception as e:
        return e

@app.post("/signup")
async def signup(data:Dict[str,List[str]]=Body(...)):
    try:
        # print(url_list)
        identifier = list(data.keys())[0]
        url_list = data[identifier]
        get_facial_encodings(identifier,url_list)  
        return {"Response": "User information saved"}
    except Exception as e:
        return {"Response": e}


@app.post("/gesture_check")
async def gesture_check(data:Dict[str,List[str]]=Body(...)):
    try:
        identifier = list(data.keys())[0]
        img_url = data[identifier][0]
        urllib.parse.quote(':')
        url_response = urllib.request.urlopen(img_url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, -1)
        image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        response = Gestures_Check(image_rgb,identifier)
        return response
    except Exception as e:
        return e
@app.get("/")
async def test():
	return "hello g"

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)
