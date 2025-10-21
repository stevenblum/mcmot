import sys
import os
from MCMOTracker import *
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

mp = ModelPlus(MODEL_PATH, "botsort.yaml")
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

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(TARGET_PATH, fourcc, fps, (width, height))

fps_counter = util.FrameRateCounter(update_interval=60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mp.detect_track_and_annotate(frame)
    fps_counter.tick()
    
    # Draw FPS on frame
    fps_counter.annotate_frame(mp.frame_annotated)
    
    out.write(mp.frame_annotated)
    cv2.imshow("Tracked Video", mp.frame_annotated)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
print(f"Saved tracked video to: {TARGET_PATH}")
print(f"Average processing FPS: {fps_counter.get_average_fps():.2f}")