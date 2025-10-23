import time
import cv2
import numpy as np
import os

# Define chessboard dimensions
CAMERA_DEVIECE_ID = 0  # Change if you have multiple cameras
CAMERA_NAME = "lab_logitech_1"

###############
OUTPUT_FILE = os.path.abspath(f"./config/camera_calibration/{CAMERA_NAME}/{CAMERA_NAME}.npz")
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
#######################################################################################################

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Open live camera feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit to stop capturing images and calculate the calibration parameters.")
last_capture_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
        if time.time() - last_capture_time > 1:  # Capture at most one frame per second
            last_capture_time = time.time()
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print("Frame captured for calibration.")

    cv2.imshow('Live Camera', frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

print(f"{len(objpoints)} frames captured for calibration.")

print("Calibrating camera...")
if objpoints and imgpoints:
    # Perform camera calibration
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibration successful!")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)

        # Save the results in the same file at OUTPUT_FILE
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        np.savez(OUTPUT_FILE, mtx=mtx, dist=dist)
        print(f"Calibration data saved to {OUTPUT_FILE}.")
    else:
        print("Camera calibration failed.")
else:
    print("Not enough data for calibration.")
