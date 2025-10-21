import cv2
import os
import numpy as np
from ultralytics import YOLO
import sys



# Example usage
if __name__ == "__main__":
    # Initialize cameras
    cap1 = cv2.VideoCapture(4)  # First camera
    cap2 = cv2.VideoCapture(6)  # Second camera
    
    # Set camera properties
    for cap in [cap1, cap2]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Calibrate
    calibration = calibrate_cameras_with_chessboard(cap1, cap2)
    
    if calibration:
        # Save calibration data
        np.savez('camera_calibration.npz', **calibration)
        print("Calibration saved to camera_calibration.npz")
        homographies = [np.eye(3), calibration['homography']]
    else:
        print("Calibration failed or was not performed.")
        exit()
    
    # Check training results before loading model
    check_training_results()
    
    model_weights_path = find_latest_model()[0]
    detector = YOLO(model_weights_path) 
    detector.conf = .05
    detector.classes = [0, 1, 2, 3, 4, 5]
    
    # Debug: Print model information
    print(f"Model class names: {detector.names}")
    print(f"Number of classes: {len(detector.names)}")
    expected_classes = ['red_square', 'red_circle', 'green_square', 'green_circle', 'blue_square', 'blue_circle']
    for i, expected in enumerate(expected_classes):
        actual = detector.names.get(i, 'MISSING')
        status = "✓" if actual == expected else "✗"
        print(f"  Class {i}: {status} Expected '{expected}', Got '{actual}'")

    trackers = [sort.Sort(max_age=30, min_hits=3, iou_threshold=0.3) for _ in range(2)]
    
    try:
        global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)
    except AttributeError:
        print("Error: MultiCameraTracker class not found in homography_tracker module.")
        print("Please check if the class exists or implement it.")
        sys.exit(1)

    while True:
        # Get frames
        frame1 = cap1.read()[1]
        frame2 = cap2.read()[1]

        # Convert to RGB for YOLO detection
        rgb_frame1 = frame1[:, :, ::-1]
        rgb_frame2 = frame2[:, :, ::-1]
        frames_rgb = [rgb_frame1, rgb_frame2]

        # Run object detection
        results = detector(frames_rgb)

        dets, tracks = [], []
        
        for i, result in enumerate(results):
            print(f"Camera {i}:")
            # Extract detections from result object
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                det = []
                
                # Convert boxes to numpy format
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class indices
                    
                    for j in range(len(xyxy)):
                        class_id = int(cls[j])
                        class_name = detector.names.get(class_id, f"unknown_class_{class_id}")
                    
                    # Combine into detection format: [x1, y1, x2, y2, conf, cls]
                    for j in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[j]
                        det.append([x1, y1, x2, y2, conf[j], cls[j]])
                    
                    det = np.array(det)
                else:
                    det = np.empty((0, 6))  # Empty array with correct shape
            else:
                det = np.empty((0, 6))  # Empty array if no detections
            
            dets.append(det)

            # Updating each tracker - convert to format expected by SORT
            if len(det) > 0:
                # SORT expects [x1, y1, x2, y2, score] format
                sort_dets = det[:, :5]  # Take first 5 columns
                tracker_result = trackers[i].update(sort_dets)
            else:
                tracker_result = np.empty((0, 5))  # Empty tracking result
            
            tracks.append(tracker_result)

        # Updating the global tracker
        global_ids = global_tracker.update(tracks)

        # Draw tracks on BGR frames (original camera frames)
        frames_bgr = [frame1.copy(), frame2.copy()]
        for i in range(2):
            if i < len(tracks) and i < len(global_ids):
                frames_bgr[i] = utilities.draw_tracks(
                    frames_bgr[i], tracks[i], global_ids[i], i, classes=detector.names
                )

        # Combine frames for display
        vis = np.hstack(frames_bgr)

        cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
        cv2.imshow("Vis", vis)
        key = cv2.waitKey(1)  # Changed from waitKey(0) to waitKey(1) for real-time display

        if key == ord("q"):
            break
        
    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()