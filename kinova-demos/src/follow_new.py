import sys
import collections
import collections.abc
# Monkey patch for Python 3.10+
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    collections.MutableMapping = collections.abc.MutableMapping
    collections.MutableSequence = collections.abc.MutableSequence

import numpy as np
import cv2
import time
import os
import argparse
import wget
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner

from videocaptureasync import VideoCaptureAsync
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

# --- Face Tracker Class ---
class FaceTracker:
    """
    A simple tracker for faces based on centroid tracking and disappearance count.
    """
    def __init__(self, max_disappeared=30):
        self.next_face_id = 0
        self.faces = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        self.faces[self.next_face_id] = {'centroid': centroid, 'box': box}
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1

    def deregister(self, face_id):
        del self.faces[face_id]
        del self.disappeared[face_id]

    def update(self, rects):
        if len(rects) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            return self.faces

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_boxes = []
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)
            input_boxes.append((startX, startY, endX, endY))

        if len(self.faces) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], input_boxes[i])
        else:
            face_ids = list(self.faces.keys())
            face_centroids = np.array([f['centroid'] for f in self.faces.values()])
            D = dist.cdist(face_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                face_id = face_ids[row]
                self.faces[face_id] = {'centroid': input_centroids[col], 'box': input_boxes[col]}
                self.disappeared[face_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                face_id = face_ids[row]
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            for col in unused_cols:
                self.register(input_centroids[col], input_boxes[col])
        return self.faces

# --- Robot Control Functions ---
def setup_robot_connection(ip, port, username, password):
    """Establishes connection with the robot and returns the base client."""
    transport = TCPTransport()
    router = RouterClient(transport, lambda kException: print("Kortex API Error:", kException))
    transport.connect(ip, port)

    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = username
    session_info.password = password
    session_info.session_inactivity_timeout = 60000
    session_info.connection_inactivity_timeout = 2000

    session_manager = SessionManager(router)
    session_manager.CreateSession(session_info)

    base_client = BaseClient(router)
    return base_client, session_manager, transport, router

def send_home(base_client):
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base_client.ReadAllActions(action_type)
    action_handle = next((action.handle for action in action_list.action_list if action.name == "Home"), None)
    if action_handle:
        base_client.ExecuteActionFromReference(action_handle)
        time.sleep(6) # Give time for the action to complete

def gripper_command(base_client, value):
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.value = value # 0 is open, 1 is closed
    base_client.SendGripperCommand(gripper_command)

def twist_command(base_client, cmd):
    command = Base_pb2.TwistCommand()
    command.duration = 0
    twist = command.twist
    twist.linear_x, twist.linear_y, twist.linear_z, twist.angular_x, twist.angular_y, twist.angular_z = cmd
    base_client.SendTwistCommand(command)

# --- UI and Detection Functions ---
def create_layout() -> Layout:
    layout = Layout(name="root")
    layout.split(Layout(name="header", size=3), Layout(name="info", ratio=1), Layout(name="footer", size=3))
    return layout

def create_info_table(stats: dict) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="magenta")

    table.add_row("FPS:", f"{stats.get('fps', 0):.2f}")
    table.add_row("Source:", stats.get('source', 'N/A'))
    table.add_row("Faces Detected:", str(stats.get('face_count', 0)))
    main_face_id = stats.get('main_face_id')
    table.add_row("Main Face ID:", Text(str(main_face_id), style="bold green") if main_face_id is not None else "N/A")
    table.add_row("Confidence:", f"{stats.get('confidence', 0.4):.2f}")
    table.add_row("Velocity:", f"{stats.get('velocity', 1.0):.2f}x")
    table.add_row("Robot Status:", stats.get('robot_status', Text("Disconnected", style="red")))
    cmd_str = ", ".join([f"{c:.2f}" for c in stats.get('robot_cmd', [0]*6)])
    table.add_row("Robot Command:", cmd_str)
    return table

def find_all_faces(frame: np.ndarray, detector: YOLO, factor: float, confidence: float):
    results = detector(frame, verbose=False, conf=confidence)
    boxes = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            bbx = box.xyxy.cpu().numpy()[0]
            boxes.append(tuple(int(c / factor) for c in bbx))
    return boxes

def setup_detector() -> YOLO:
    model_path = "assets/yolov8n-face.pt"
    os.makedirs("assets", exist_ok=True)
    console = Console()
    if not os.path.exists(model_path):
        with console.status("[bold green]Downloading YOLOv8 face model...", spinner="dots"):
            wget.download("https://github.com/verlab/demos-verlab/releases/download/kinova/yolov8n-face.pt", model_path)
    with console.status("[bold green]Loading YOLOv8 model...", spinner="dots"):
        detector = YOLO(model_path)
    return detector

def main(args):
    console = Console()
    detector = setup_detector()
    tracker = FaceTracker()
    
    # --- Robot Setup ---
    base_client, session_manager, transport, router = None, None, None, None
    try:
        raise NotImplementedError("Robot connection setup is not implemented in this demo.")
        base_client, session_manager, transport, router = setup_robot_connection(args.robot_ip, 10000, 'admin', 'admin')
        robot_status = Text("Connected", style="green")
        send_home(base_client)
    except Exception as e:
        console.print(f"[bold red]Robot connection failed: {e}[/bold red]")
        robot_status = Text("Connection Failed", style="red")

    # --- Video and UI Setup ---
    video_capture = VideoCaptureAsync(args.source)
    video_capture.start()
    time.sleep(1.0) # Allow camera to start
    
    ret, frame = video_capture.read()
    if not ret:
        console.print("[bold red]Could not read from video source. Exiting.[/bold red]")
        return

    RESIZE_FACTOR = 0.5
    stats = {
        'fps': 0, 'source': args.source,
        'width': frame.shape[1], 'height': frame.shape[0],
        'face_count': 0, 'main_face_id': None,
        'confidence': args.confidence,
        'velocity': args.velocity,
        'robot_status': robot_status, 'robot_cmd': [0]*6
    }
    
    layout = create_layout()
    layout["header"].update(Panel(Text("Robot Face Tracking Demo", justify="center", style="bold green"), border_style="green"))
    layout["footer"].update(Panel(Text("a/d: Face | w/s: Conf | r/f: Velocity | h: Home | o/c: Gripper | q: Quit", justify="center", style="yellow")))

    main_face_idx = 0

    with Live(layout, console=console, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
        prev_frame_time = time.time()
        while True:
            ret, frame = video_capture.read()
            if not ret: continue

            center_X, center_Y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
            
            small_frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            face_boxes = find_all_faces(small_frame, detector, RESIZE_FACTOR, stats['confidence'])
            tracked_faces = tracker.update(face_boxes)
            
            stats['face_count'] = len(tracked_faces)
            tracked_ids = sorted(list(tracked_faces.keys()))
            cmd = np.zeros(6)

            if not tracked_ids:
                stats['main_face_id'] = None
            else:
                if main_face_idx >= len(tracked_ids): main_face_idx = 0
                stats['main_face_id'] = tracked_ids[main_face_idx]

                main_face_data = tracked_faces.get(stats['main_face_id'])
                if main_face_data:
                    box = main_face_data['box']
                    face_center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                    
                    # --- Robot Movement Calculation ---
                    if base_client:
                        dx = face_center[0] - center_X
                        dy = face_center[1] - center_Y
                        
                        # Angular velocity (look left/right) based on horizontal distance
                        cmd[4] = -np.clip(dx * 0.01 * stats['velocity'], -0.8, 0.8) 
                        # Angular velocity (look up/down) based on vertical distance
                        cmd[3] = np.clip(dy * 0.01 * stats['velocity'], -0.8, 0.8)
            
            if base_client:
                twist_command(base_client, list(cmd))
                stats['robot_cmd'] = cmd

            for (face_id, data) in tracked_faces.items():
                box = data['box']
                color = (0, 255, 0) if face_id == stats['main_face_id'] else (255, 165, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"ID {face_id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            new_frame_time = time.time()
            stats['fps'] = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            layout["info"].update(Panel(create_info_table(stats), title="[b]Real-Time Info[/b]", border_style="blue"))
            cv2.imshow('Video Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('w'): stats['confidence'] = min(stats['confidence'] + 0.05, 0.95)
            elif key == ord('s'): stats['confidence'] = max(stats['confidence'] - 0.05, 0.05)
            elif key == ord('r'): stats['velocity'] = min(stats['velocity'] + 0.1, 5.0)
            elif key == ord('f'): stats['velocity'] = max(stats['velocity'] - 0.1, 0.1)
            elif key == ord('h') and base_client: send_home(base_client)
            elif key == ord('o') and base_client: gripper_command(base_client, 0) # Open
            elif key == ord('c') and base_client: gripper_command(base_client, 1) # Close
            
            if tracked_ids:
                if key == ord('a'): main_face_idx = (main_face_idx - 1 + len(tracked_ids)) % len(tracked_ids)
                elif key == ord('d'): main_face_idx = (main_face_idx + 1) % len(tracked_ids)

    # --- Cleanup ---
    video_capture.stop()
    cv2.destroyAllWindows()
    if base_client:
        send_home(base_client)
        console.print('Closing Robot Session...')
        session_manager.CloseSession()
        router.SetActivationStatus(False)
        transport.disconnect()
    console.print("[bold blue]Application terminated.[/bold blue]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Face Tracking Demo.")
    parser.add_argument("--source", type=str, default="rtsp://150.164.212.190/color", help="Video source (RTSP URL or file path).")
    parser.add_argument("--confidence", type=float, default=0.4, help="Initial confidence threshold.")
    parser.add_argument("--robot_ip", type=str, default="150.164.212.190", help="IP address of the Kinova robot.")
    parser.add_argument("--velocity", type=float, default=1.0, help="Initial velocity multiplier for robot movement.")
    args = parser.parse_args()
    main(args)
