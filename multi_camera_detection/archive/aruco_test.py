import cv2
import cv2.aruco as aruco
import numpy as np
import time
from util import select_cameras

def main():
    selected = select_cameras(10)
    print(f"In Aruco Test, Selected camera: {selected}")
    if not selected:
        print("No cameras selected.")
        return

    cam_idx = selected[0]
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open camera {cam_idx}")
        return

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters()

    last_look_homography_time = 0
    last_compute_homography_time = 0
    H = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        # Draw and label all detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                c = corner[0]
                center = c.mean(axis=0).astype(int)
                cv2.putText(frame, f"ID:{ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                pt1 = tuple(c[0].astype(int))
                pt2 = tuple(c[1].astype(int))
                cv2.arrowedLine(frame, pt1, pt2, (255,0,0), 2, tipLength=0.3)
        cv2.putText(frame, f"Last Homography:{last_compute_homography_time:.2f}", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Every 1 second, try to compute homography from markers 9, 14, 19
        now = time.time()
        if now - last_look_homography_time > 1.0 and ids is not None:
            last_look_homography_time = now
            marker_ids = [9, 14, 19]
            img_pts = []
            world_pts = []
            for i, marker_id in enumerate(marker_ids):
                idx = np.where(ids.flatten() == marker_id)[0]
                if len(idx) > 0:
                    # Use the first corner of the marker as the reference point
                    # We'll use the center for better stability
                    c = corners[idx[0]][0]
                    center = c.mean(axis=0)
                    img_pts.append(center)
                    if marker_id == 9:
                        world_pts.append([0, 0])
                    elif marker_id == 14:
                        world_pts.append([12, 0])
                    elif marker_id == 19:
                        world_pts.append([0, 12])
            if len(img_pts) == 3:
                last_compute_homography_time = now
                img_pts = np.array(img_pts, dtype=np.float32)
                world_pts = np.array(world_pts, dtype=np.float32)
                # Add a fourth point for homography (simulate a rectangle)
                # Estimate the fourth world point (12,12)
                img_vec = img_pts[1] - img_pts[0] + img_pts[2] - img_pts[0]
                img_pts4 = np.vstack([img_pts, img_pts[0] + img_vec])
                world_pts4 = np.vstack([world_pts, [12,12]])
                H, status = cv2.findHomography(img_pts4, world_pts4)
                if H is not None:
                    print("[INFO] Homography matrix computed.")
                else:
                    print("[WARN] Homography computation failed.")


        # If homography is available, look for marker 25 and report its position
        if H is not None and ids is not None:
            idx25 = np.where(ids.flatten() == 25)[0]
            if len(idx25) > 0:
                c25 = corners[idx25[0]][0]
                center25 = c25.mean(axis=0)
                pt = np.array([[center25]], dtype=np.float32)
                world_pt = cv2.perspectiveTransform(pt, H)[0][0]
                cv2.putText(frame, f"25: ({world_pt[0]:.1f},{world_pt[1]:.1f})", tuple(center25.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                print(f"[INFO] Marker 25 position: ({world_pt[0]:.2f}, {world_pt[1]:.2f})")

        cv2.imshow(f"Aruco Camera {cam_idx}", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
