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
        """Register a new face with a new ID."""
        self.faces[self.next_face_id] = {'centroid': centroid, 'box': box}
        self.disappeared[self.next_face_id] = 0
        self.next_face_id += 1

    def deregister(self, face_id):
        """Deregister a face that has disappeared."""
        del self.faces[face_id]
        del self.disappeared[face_id]

    def update(self, rects):
        """
        Update the tracker with a new set of detected bounding boxes.
        """
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
                if row in used_rows or col in used_cols:
                    continue
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

# --- UI and Detection Functions ---

def create_layout() -> Layout:
    """Defines the simplified terminal layout."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="info", ratio=1),
        Layout(name="footer", size=3),
    )
    return layout

def create_info_table(stats: dict) -> Table:
    """Creates a table with real-time stats."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="magenta")

    table.add_row("FPS:", f"{stats.get('fps', 0):.2f}")
    table.add_row("Source:", stats.get('source', 'N/A'))
    table.add_row("Resolution:", f"{int(stats.get('width', 0))}x{int(stats.get('height', 0))}")
    table.add_row("Faces Detected:", str(stats.get('face_count', 0)))
    
    main_face_id = stats.get('main_face_id')
    if main_face_id is not None:
        table.add_row("Main Face ID:", Text(str(main_face_id), style="bold green"))
    else:
        table.add_row("Main Face ID:", "N/A")
    
    table.add_row("Confidence:", f"{stats.get('confidence', 0.4):.2f}")
        
    return table

def find_all_faces(frame: np.ndarray, detector: YOLO, factor: float, confidence: float):
    """Detects all faces in a frame based on a confidence threshold."""
    results = detector(frame, verbose=False, conf=confidence)
    boxes = []
    if len(results) > 0 and len(results[0].boxes) > 0:   
        for box in results[0].boxes:
            bbx = box.xyxy.cpu().numpy()[0]
            boxes.append(tuple(int(c / factor) for c in bbx))
    return boxes

def setup_detector() -> YOLO:
    """Downloads the model if it doesn't exist and loads it."""
    model_path = "assets/yolov8n-face.pt"
    os.makedirs("assets", exist_ok=True)
    
    console = Console()
    if not os.path.exists(model_path):
        with console.status("[bold green]Downloading YOLOv8 face model...", spinner="dots"):
            wget.download("https://github.com/verlab/demos-verlab/releases/download/kinova/yolov8n-face.pt", model_path)
            console.log(f"Model downloaded to {model_path}")

    with console.status("[bold green]Loading YOLOv8 model...", spinner="dots"):
        detector = YOLO(model_path)
        console.log("Model loaded successfully.")
    return detector

def main(args):
    """Main function to run the face detection demo."""
    console = Console()
    detector = setup_detector()
    tracker = FaceTracker()
    
    if "rtspsrc" in args.source:
        video_capture = cv2.VideoCapture(args.source, cv2.CAP_GSTREAMER)
    else:
        video_capture = cv2.VideoCapture(args.source)

    if not video_capture.isOpened():
        console.print(f"[bold red]Error: Could not open video source: {args.source}[/bold red]")
        return

    RESIZE_FACTOR = 0.5
    stats = {
        'fps': 0, 'source': args.source,
        'width': video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'face_count': 0, 'main_face_id': None,
        'confidence': args.confidence
    }
    
    layout = create_layout()
    layout["header"].update(Panel(Text("Face Detection & Tracking Demo", justify="center", style="bold green"), border_style="green"))
    layout["footer"].update(Panel(Text("a/d: Switch Face | w/s: Adjust Confidence | q: Quit", justify="center", style="yellow"), border_style="yellow"))

    main_face_idx = 0

    with Live(layout, console=console, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
        prev_frame_time = time.time()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                if "rtsp" not in args.source:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    console.log("[bold red]Stream ended or could not read frame.[/bold red]")
                    break
            
            small_frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            face_boxes = find_all_faces(small_frame, detector, RESIZE_FACTOR, stats['confidence'])
            tracked_faces = tracker.update(face_boxes)
            
            stats['face_count'] = len(tracked_faces)
            tracked_ids = sorted(list(tracked_faces.keys()))

            if not tracked_ids:
                stats['main_face_id'] = None
            else:
                if main_face_idx >= len(tracked_ids):
                    main_face_idx = 0
                stats['main_face_id'] = tracked_ids[main_face_idx]

            for (face_id, data) in tracked_faces.items():
                box = data['box']
                color = (0, 255, 0) if face_id == stats['main_face_id'] else (255, 165, 0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                text = f"ID {face_id}"
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            new_frame_time = time.time()
            stats['fps'] = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            info_panel = Panel(create_info_table(stats), title="[b]Real-Time Info[/b]", border_style="blue")
            layout["info"].update(info_panel)
            
            cv2.imshow('Video Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # --- Adjust Confidence ---
            if key == ord('w'):
                stats['confidence'] = min(stats['confidence'] + 0.05, 0.95)
            elif key == ord('s'):
                stats['confidence'] = max(stats['confidence'] - 0.05, 0.05)

            # --- Switch Main Face ---
            if tracked_ids:
                if key == ord('a'): 
                    main_face_idx = (main_face_idx - 1 + len(tracked_ids)) % len(tracked_ids)
                elif key == ord('d'):
                    main_face_idx = (main_face_idx + 1) % len(tracked_ids)

    video_capture.release()
    cv2.destroyAllWindows()
    console.print("[bold blue]Application terminated.[/bold blue]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection Demo with Rich TUI and Tracking.")
    parser.add_argument("--source", type=str, default="./example.mp4", help="Video source (file path or RTSP URL).")
    parser.add_argument("--confidence", type=float, default=0.2, help="Initial confidence threshold for face detection.")
    args = parser.parse_args()

    if args.source == "./example.mp4" and not os.path.exists("./example.mp4"):
        print("Default 'example.mp4' not found. Please provide a video source using --source")
    else:
        main(args)
