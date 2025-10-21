import cv2
import os


VIDEO_PATH = "/home/scblum/Projects/testbed_cv/raw_images/2025 10 15 Hanoi3 Fixed/3 disks.mp4"
OUTPUT_FOLDER = "/home/scblum/Projects/testbed_cv/raw_images/2025 10 15 Hanoi3 Fixed"

VIDEO_PATH = "/home/scblum/Projects/testbed_cv/raw_images/2025 06 01 testbed simulation/best_video_normal_segments/segment_005/best_video_normal_segment_005.mp4"
OUTPUT_FOLDER = os.path.splitext(VIDEO_PATH)[0]
IMAGE_PREFIX = os.path.basename(VIDEO_PATH).split('.')[0]
FRAME_RATE=5


# Create the output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit(1)

# Get original video FP
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(video_fps / FRAME_RATE)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every nth frame based on desired frame rate
    if count % frame_interval == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"{IMAGE_PREFIX}_frame_{saved:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()
print(f"Saved {saved} frames to '{OUTPUT_FOLDER}'")