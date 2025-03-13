"""Terminal command:

color: gst-launch-1.0 rtspsrc location=rtsp://192.168.1.10/color latency=30 ! rtph264depay ! avdec_h264 ! autovideosink

depth: gst-launch-1.0 rtspsrc location=rtsp://192.168.1.10/depth latency=30 ! rtpgstdepay ! videoconvert ! autovideosink
"""

import numpy as np
import cv2
# import dlib
# from imutils import face_utils
import time
from ultralytics import YOLO

from fast_rtsp import RTSPCapture
import wget
import os

def find_one_face(frame, detector, FACTOR):
    results = detector(frame)
    if len(results) > 0:
        if len(results[0].boxes) > 0:   
            bbx = results[0].boxes[0].xyxy.cpu().numpy()[0]
            return int(bbx[0] / FACTOR), int(bbx[1] / FACTOR), int(bbx[2] / FACTOR), int(bbx[3] / FACTOR)
    return None

if __name__ == "__main__":
    
    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture("rtsp://192.168.1.10/color")
    # video_capture = RTSPCapture("rtsp://192.168.1.10/color")
    # detector = dlib.get_frontal_face_detector()
    os.makedirs("assets", exist_ok=True)
    wget.download("https://github.com/verlab/demos-verlab/releases/download/kinova/yolov8n-face.pt", "assets/yolov8n-face.pt")
    
    detector = YOLO("assets/yolov8n-face.pt")
    FACTOR = 0.3


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        start = time.time()

        res = cv2.resize(frame, None, fx=FACTOR, fy=FACTOR)
        # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # rects = detector(gray)    
        # results = detector(res)
        
        # for (i, res) in enumerate(results):
        #     bbx = res.boxes[0].xyxy.cpu().numpy()[0]
        #     conf = res.boxes[0].conf
        #     if conf > 0.5:
        #         x1 = int(bbx[0] / FACTOR)
        #         y1 = int(bbx[1] / FACTOR)
        #         x2 = int(bbx[2] / FACTOR)
        #         y2 = int(bbx[3] / FACTOR)
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        face = find_one_face(res, detector, FACTOR)
        if face:
            cv2.rectangle(frame, face[:2], face[2:], (0, 255, 0), 2)

        # Display the video output
        cv2.imshow('Video', frame)

        # Quit video by typing Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end = time.time()

        print(chr(27) + "[2J")
        print("FPS: {}".format(1/(end - start)))


    video_capture.release()
    cv2.destroyAllWindows()

