import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util

import supervision as sv
from ultralytics import YOLO
import cv2



#VIDEO_PATH = "/home/scblum/Projects/testbed_cv/raw_images/2025 10 15 Hanoi4 Fixed/4 disks.mp4"
VIDEO_PATH = "/home/scblum/Projects/testbed_cv/raw_images/2025 06 01 testbed simulation/best_video_normal.mp4"


MODEL_PATH = util.find_latest_model()[0]

# Make target by adding "_tracked" before the file extension
root_path = os.path.dirname(VIDEO_PATH)
base, ext = os.path.splitext(os.path.basename(VIDEO_PATH))
TARGET_PATH = os.path.join(root_path, f'{base}_tracked{ext}')

################################################################################################################

# Get video info using OpenCV directly
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Cannot open video at: {VIDEO_PATH}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=total_frames)
print(f"Video info: {video_info}")

model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack(frame_rate=video_info.fps)

smoother = sv.DetectionsSmoother()

colors = sv.ColorPalette.from_hex(["#ff0303",  "#ef16ae", "#0F8807", "#4af84a", "#003CFF", "#49d2f8"])
annotator = sv.BoxAnnotator(color=colors, thickness=2)
annotator_thin = sv.BoxAnnotator(color=colors, thickness=1)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_PATH, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=0.05, iou=.7, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections_tracker = tracker.update_with_detections(detections)
    detections_smoothed = smoother.update_with_detections(detections_tracker)

    annotated_frame = annotator.annotate(frame.copy(), detections_smoothed)
    annotated_frame = annotator_thin.annotate(annotated_frame, detections)
    out.write(annotated_frame)
    cv2.imshow("Tracked Video", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
print(f"Saved tracked video to: {TARGET_PATH}")