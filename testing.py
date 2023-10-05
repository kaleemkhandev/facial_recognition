import cv2
import argparse
from main import run



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--vid_path", help="path of the video which you want to test")
    args = argParser.parse_args()

    vid_file_path = args.vid_path
    video_capture = cv2.VideoCapture(vid_file_path)
    response= run(video_capture)
    print(response)