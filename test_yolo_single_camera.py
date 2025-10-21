import cv2
import os
import numpy as np
import time
from ultralytics import YOLO
import sys
import supervision as sv
import sys
import os
from MCMOTracker import *
import util


##############################################################

#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_13_11_20/weights/best.pt"   # or specify a path
#MODEL_PATH = "/home/scblum/Projects/testbed_cv/train_yolo/saved_models/part_detection_25_10_07_14_28/weights/best.pt"   # or specify a path
MODEL_PATH = "LATEST"
ARUCO_POSITIONS = "square4"  #  None
CONFIDENCE_THRESHOLD = 0.05

CAMERA_DEVICE_ID = 4  # v4l2-ctl --list-devices
CAMERA_NAME = "logitech_1"

##############################################################
if MODEL_PATH == "LATEST":
    MODEL_PATH= util.find_latest_model()[0]

frc = util.FrameRateCounter(update_interval=60)
if __name__ == "__main__":
    cam = Camera(0, CAMERA_DEVICE_ID, CAMERA_NAME, MODEL_PATH, tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
    while cam.cap.isOpened():
        ret, frame = cam.cap.read()
        if not ret:
            break

        cam.detect_track_and_annotate()

        frc.tick()
        frc.annotate_frame(cam.model_plus.frame_annotated)

        cv2.imshow("Tracked Video", cam.model_plus.frame_annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.cap.release()
    cv2.destroyAllWindows()
