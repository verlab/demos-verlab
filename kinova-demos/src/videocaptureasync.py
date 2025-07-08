# file: videocaptureasync.py
import threading
import cv2
from fast_rtsp import RTSPCapture
import os

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, use_rtsp=False):
        self.src = src
        use_rtsp = use_rtsp or (src.startswith("rtsp://") or src.startswith("rtmp://"))
        
        # if src is just number or a string that can be converted to int, treat it as a camera index
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            self.src = int(src)
            use_rtsp = False
        
        if use_rtsp:
            self.cap = RTSPCapture(self.src)
        elif os.path.isfile(self.src):
            self.cap = cv2.VideoCapture(self.src)
        else:
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.src}")
        else:
            print(f"Video source {self.src} opened successfully.")

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                # reset the video capture if reading fails
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                grabbed, frame = self.cap.read()
            if not grabbed:
                print("[!] Failed to grab frame from video source.")
                continue
            
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()