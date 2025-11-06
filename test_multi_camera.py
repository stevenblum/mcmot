from mcmot import MCMOTracker, MCMOTUtils
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2


'''
#LAB
MODEL_PATH = util.find_latest_model()[0]
CAMERA_DEVICE_IDS = [0,6]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
'''

# HOME

MODEL_PATH = "LATEST"
CAMERA_DEVICE_IDS = "SELECT"  # [4,6]  # v4l2-ctl --list-devices
CAMERA_NAMES = ["logitech_1","logitech_1"]
ARUCO_POSITIONS = "square4"  #  None
###################################################################

if MODEL_PATH == "LATEST":
    MODEL_PATH= MCMOTUtils.find_latest_model()[0]

if CAMERA_DEVICE_IDS == "SELECT":
    CAMERA_DEVICE_IDS =  MCMOTUtils.get_camera_number(num_cameras=2)

mct = MCMOTracker(MODEL_PATH,CAMERA_DEVICE_IDS,CAMERA_NAMES,confidence_threshold=.05,tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
mct.select_cameras()

while True:
    mct.capture_cameras_frames()
    mct.update_cameras_tracks()
    mct.match_global_tracks()

    for camera_number in range(len(mct.cameras)):
        frame = mct.annotated_frame_from_camera(camera_number)
        overhead = mct.plot_overhead()
        
        if frame is not None:
            cv2.imshow(f"Camera {camera_number}", frame)

        cv2.imshow("Overhead", overhead)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
