from MCMOTracker import *
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import cv2
import tkinter as tk
root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()

#LAB
CAMERA_DEVICE_IDS = [0,1]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["lab_logitech_1","lab_logitech_2"]

# HOME
'''
CAMERA_DEVICE_IDS = [0,1]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["lab_logitech_1","lab_logitech_2"]
'''

##############
# Common 
MODEL_PATH = "LATEST"
MODEL_PATH = os.path.relpath("saved_models/part_detection_25_10_16_00_57/part_detection_25_10_16_00_57/weights/best.pt")
ARUCO_POSITIONS = "square4"  #  None

###################################################################

if MODEL_PATH == "LATEST":
    MODEL_PATH= util.find_latest_model()[0]

mct = MCMOTracker(MODEL_PATH,CAMERA_DEVICE_IDS,CAMERA_NAMES,tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
mct.select_cameras()
cv2.namedWindow('Camera 0', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera 1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera 0', SCREEN_WIDTH//2, SCREEN_HEIGHT//2)

while True:
    mct.update_camera_tracks()
    mct.match_global_tracks()
    frame0 = mct.frame_from_cam(0)
    frame1 = mct.frame_from_cam(1)
    
    if frame0 is not None:
        cv2.imshow("Camera 0", frame0)
    if frame1 is not None:
        cv2.imshow("Camera 1", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
