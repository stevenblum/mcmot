from .ModelPlus import ModelPlus
from . import MCMOTUtils
import time
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from config.aruco_config import ArucoConfig

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
detector = cv2.aruco.ArucoDetector(aruco_dict)

class Camera:    
    def __init__(self, camera_number, camera_device_id, camera_name, model_path, confidence_threshold, tracker_yaml_path, aruco_positions):
        self.camera_number = camera_number
        self.camera_device_id = camera_device_id
        print(f"    Opening Camera on Port {camera_device_id}")
        self.cap = cv2.VideoCapture(camera_device_id)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps_counter = MCMOTUtils.FrameRateCounter()
        self.frame = None
        self.frame_dims = None #(width, height)
        self.frame_time = None
        self.frame_annotated = None
        self.camera_name = camera_name
        self.mtx = None
        self.dist = None
        self.world_calibration_status = False
        self.world_axis_c = None
        self.fov_ground_pts = None

        self.model_plus = ModelPlus(model_path, confidence_threshold, tracker_yaml_path)
        self.set_intrinsics()

        if aruco_positions == None:
            self.aruco_positions = None
        else:
            self.rvec=None
            self.tvec=None
            self.Rt = None
            self.aruco_positions = aruco_positions
            self.aruco_config = ArucoConfig("square4")
            self.set_extrinsics()
            self.world_calibration_status = True
            self.compute_fov_on_ground()

    def set_intrinsics(self):
        calibration_data = np.load(f"./config/camera_calibration/{self.camera_name}/{self.camera_name}.npz")
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']

    def set_extrinsics(self):
        # Show Camera To Make Sure ArUco Markers are Visible
        while True:
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.frame_dims is None:
                self.frame_dims = np.shape(gray)
            corners, ids, rejected = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    c = corner[0]
                    center = c.mean(axis=0).astype(int)
                    cv2.putText(frame, f"ID:{ids[i][0]}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    pt1 = tuple(c[0].astype(int))
                    pt2 = tuple(c[1].astype(int))
                    cv2.arrowedLine(frame, pt1, pt2, (255,0,0), 2, tipLength=0.3)
            cv2.imshow(f"Camera {self.camera_name} - Place ArUco Markers in View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        aruco_detections = []
        # Capture the corner of the ArUco markers in 10 frames
        for t in range(10):
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(gray)
                if ids is not None:
                    for id, corner in zip(ids.flatten(), corners):
                        aruco_detections.append((id,corner))
            time.sleep(0.1)

        # Find the average position of each detected ArUco marker
        aruco_positions = {}
        for id, corner in aruco_detections:
            if id in self.aruco_config.marker_positions:
                if id not in aruco_positions:
                    aruco_positions[id] = []
                aruco_positions[id].append(corner)
        for id in aruco_positions:
            aruco_positions[id] = np.mean(aruco_positions[id], axis=0)

        # Prepare image and object points for cv2.solvePnP
        object_points = []
        image_points = []
        for id, corner in aruco_positions.items():
            if id in self.aruco_config.marker_positions:
                object_points.append(self.aruco_config.marker_positions[id])
                image_points.append(np.mean(corner[0], axis=0))  # Average corner position
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        print(object_points)
        print(image_points)

        # Solve for rotation and translation vectors with cv2.solvePnP
        if len(object_points) >= 4:  # Need at least 4 points to solve PnP
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, self.mtx, self.dist)
            self.rvec = rvec
            self.r = cv2.Rodrigues(rvec)[0]
            self.tvec = tvec
            self.t = tvec
            print(f"Camera {self.camera_name} extrinsics set.")
        else:
            print(f"Error: Not enough ArUco markers detected for camera {self.camera_name}. Need at least 4.")

        # Construct the extrinsic matrix [R|t]
        # Rodrigues converts rvec to R matrix
        Rt = np.hstack((cv2.Rodrigues(self.rvec)[0], self.tvec))

        self.Rt = Rt

        # Calculate the projection matrix
        self.projMatrix = np.dot(self.mtx, Rt)

    def capture_frame(self):
        ret, self.frame = self.cap.read()
        if ret:
            self.frame_time = time.time()

    def detect_and_track(self):
        self.fps_counter.tick()
        if time.time()-self.frame_time > .1:
            print("ERROR: COMPLETING camera.detect_track_and_annotate() on a frame that is more than .1s old")

        self.model_plus.detect_and_track(self.frame)

    def annotate_frame(self):
        self.model_plus.annotate_frame()
        af = self.model_plus.frame_annotated.copy()
        if self.world_calibration_status:
            af= self.draw_world_axis(af)
        
        fps_text = f"FPS: {self.fps_counter.get_fps():.1f}"
        af = cv2.putText(af, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0,225,0), 2)

        self.frame_annotated = af
        return af

    def draw_world_axis(self, frame):
        if self.world_axis_c == None:
            axis_length = 10  # Length of the axis lines in meters
            axis_points_w = np.float32([[0,0,0],[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
            axis_points_c, _ = cv2.projectPoints(axis_points_w, self.rvec, self.tvec, self.mtx, self.dist)
            self.world_axis_c = []
            for i in range(4):
                self.world_axis_c.append(tuple(int(x) for x in axis_points_c[i].ravel()))

        frame = cv2.line(frame, self.world_axis_c[0],  self.world_axis_c[1], (255,0,0), 3)  # X-axis in blue
        frame = cv2.line(frame, self.world_axis_c[0],  self.world_axis_c[2], (0,255,0), 3)  # Y-axis in green
        frame = cv2.line(frame, self.world_axis_c[0],  self.world_axis_c[3], (0,0,255), 3)  # Z-axis in red
        return frame

    def transform_w2c(self,points3d_world):
        return cv2.projectPoints(points3d_world, self.r, self.t, self.mtx, self.dist)[0].flatten()
    
    def compute_fov_on_ground(self, ground_z=0.0):
        """Compute and cache the field of view intersection with ground plane."""
        if self.fov_ground_pts is not None:
            return self.fov_ground_pts  # already computed

        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        width, height = self.frame_dims

        fov_x = 2 * np.arctan(width / (2 * fx))
        fov_y = 2 * np.arctan(height / (2 * fy))

        # four corner rays in camera coordinates
        corners_cam = np.array([
            [-np.tan(fov_x / 2), -np.tan(fov_y / 2), 1],  # bottom-left
            [ np.tan(fov_x / 2), -np.tan(fov_y / 2), 1],  # bottom-right
            [ np.tan(fov_x / 2),  np.tan(fov_y / 2), 1],  # top-right
            [-np.tan(fov_x / 2),  np.tan(fov_y / 2), 1],  # top-left
        ])

        # transform rays to world coordinates
        rays_world = (self.r.T @ corners_cam.T).T
        rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)

        # camera center in world coords
        C = -self.r.T @ self.t
        C = C.flatten()

        # compute intersection with z = ground_z
        pts = []
        for v in rays_world:
            if np.abs(v[2]) < 1e-6:  # nearly parallel
                continue
            s = (ground_z - C[2]) / v[2]
            if s <= 0:
                continue
            P = C + s * v
            pts.append(P)
        pts = np.array(pts)

        # choose the two ground points with largest lateral separation
        if len(pts) >= 2:
            dists = np.linalg.norm(pts[:, None, :2] - pts[None, :, :2], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            left, right = pts[i], pts[j]
        else:
            left = right = np.zeros(3)

        dist_to_origin = np.linalg.norm(C)
        left_vec = left - C
        right_vec = right - C
        left = C + left_vec / np.linalg.norm(left_vec) * dist_to_origin
        right = C + right_vec / np.linalg.norm(right_vec) * dist_to_origin

        self.fov_ground_pts = {"C": C, "left": left, "right": right} # Each is a 3D Point in World Coordinates
        return self.fov_ground_pts