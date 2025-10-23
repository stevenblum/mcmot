import cv2

def list_cameras_opencv():
    available_cameras = []
    for i in range(10):  # Check indices from 0 to 9, adjust as needed
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    camera_indices = list_cameras_opencv()
    if camera_indices:
        print("Available camera indices detected by OpenCV:")
        for index in camera_indices:
            print(f"Camera Index: {index}")
    else:
        print("No cameras detected by OpenCV.")