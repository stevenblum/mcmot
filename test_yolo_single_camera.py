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
#MODEL_PATH = "LATEST"
#MODEL_PATH = os.path.relpath("saved_models/part_detection_25_10_16_00_57/part_detection_25_10_16_00_57/weights/best.pt")
MODEL_PATH = os.path.relpath("saved_models/part_detection_25_10_16_03_30/part_detection_25_10_16_03_30/weights/best.pt")

ARUCO_POSITIONS = None #    "square4"  #  
CONFIDENCE_THRESHOLD = 0.05
CAMERA_DEVICE_ID = 0  # v4l2-ctl --list-devices
CAMERA_NAME = "lab_logitech_1"

##############################################################
if MODEL_PATH == "LATEST":
    MODEL_PATH= util.find_latest_model()[0]
    

frc = util.FrameRateCounter(update_interval=10)
if __name__ == "__main__":
    cam = Camera(0, CAMERA_DEVICE_ID, CAMERA_NAME, MODEL_PATH, tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
    cam.model_plus.model.conf = CONFIDENCE_THRESHOLD
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    while cam.cap.isOpened():
        cam.capture_frame()
        cam.detect_track_and_annotate()

        frc.tick()
        frc.annotate_frame(cam.model_plus.frame_annotated)

        cv2.imshow("Camera", cam.model_plus.frame_annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.cap.release()
    cv2.destroyAllWindows()
