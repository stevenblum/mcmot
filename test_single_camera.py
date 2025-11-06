from mcmot import Camera, MCMOTUtils
import cv2
import os

##############################################################

#MODEL_PATH = "/home/scblum/Projects/testbed_cv/saved_models/part_detection_25_10_16_00_57/weights/best.pt"  # Nano
#MODEL_PATH = "/home/scblum/Projects/testbed_cv/saved_models/part_detection_25_10_16_03_30/weights/best.pt" # Small
MODEL_PATH = "LATEST"
ARUCO_POSITIONS = None # "square4",  None
CONFIDENCE_THRESHOLD = 0.05

CAMERA_DEVICE_ID = "SELECT" # 4  # Ubuntu: v4l2-ctl --list-devices
CAMERA_NAME = "logitech_1"
DISPLAY = True

##############################################################
if MODEL_PATH == "LATEST":
    MODEL_PATH= MCMOTUtils.find_latest_model()[0]

if CAMERA_DEVICE_ID == "SELECT":
    CAMERA_DEVICE_ID =  MCMOTUtils.get_camera_number(num_cameras=1)

if DISPLAY:
    cv2.namedWindow("Tracked Video", cv2.WINDOW_NORMAL)
    # optional: set an initial size
    cv2.resizeWindow("Tracked Video", 680, 720)

if __name__ == "__main__":
    cam = Camera(0, CAMERA_DEVICE_ID, CAMERA_NAME, MODEL_PATH, confidence_threshold=.05, tracker_yaml_path="./config/bytetrack.yaml", aruco_positions=ARUCO_POSITIONS)
    
    while cam.cap.isOpened():
        cam.capture_frame()
        cam.detect_and_track()

        print(f"Frame Rate: {cam.fps_counter.fps}")

        if DISPLAY:
            af = cam.annotate_frame()
            cv2.imshow("Tracked Video", af)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.cap.release()
    cv2.destroyAllWindows()
