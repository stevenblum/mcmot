from MCMOTracker import *
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import cv2


'''
#LAB
MODEL_PATH = util.find_latest_model()[0]
CAMERA_DEVICE_IDS = [0,6]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
'''

# HOME

MODEL_PATH = "LATEST"
CAMERA_DEVICE_IDS = [4,6]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
ARUCO_POSITIONS = "square4"  #  None
###################################################################

if MODEL_PATH == "LATEST":
    MODEL_PATH= util.find_latest_model()[0]

mct = MCMOTracker(MODEL_PATH,CAMERA_DEVICE_IDS,CAMERA_NAMES,tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
mct.select_cameras()

while True:
    mct.update_camera_tracks()
    mct.match_global_tracks()
    frame = mct.frame_from_cam(0)
    
    if frame is not None:
        cv2.imshow("Camera 0", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
