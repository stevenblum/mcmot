import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import glob
import yaml
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import traceback

# Try to import supervision for enhanced NMS
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
    print("✓ Supervision library available for enhanced NMS")
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("⚠ Supervision library not available, using fallback NMS")


def find_latest_model():
    """
    Find the most recently trained YOLO model in the yolo_training directory.
    
    Returns:
        tuple: (model_path, model_name) or (None, None) if no models found
    """
    training_dir = Path("yolo_training")
    
    if not training_dir.exists():
        print("❌ yolo_training directory not found!")
        return None, None
    
    # Look for best.pt files in subdirectories
    model_files = list(training_dir.glob("*/weights/best.pt"))
    
    if not model_files:
        print("❌ No trained models found in yolo_training directory!")
        return None, None
    
    # Get the most recent model based on modification time
    latest_model = max(model_files, key=os.path.getmtime)
    
    # Extract model name from path (e.g., "part_detection_25_10_07_12_23")
    model_name = os.path.basename(os.path.dirname(os.path.dirname(latest_model)))
    
    # Get creation time for display
    mod_time = os.path.getmtime(latest_model)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
    
    print(f"🎯 Latest model found: {model_name}")
    print(f"📅 Modified: {time_str}")
    print(f"📁 Path: {latest_model}")
    
    return str(latest_model), model_name

def load_detection_config(config_path="detection_config.yml"):
    """
    Load detection configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠ Configuration file not found: {config_path}")
        print("Using default configuration...")
        # Return default configuration
        return {
            'cameras': {'indices': [4, 6], 'resolution': {'width': 640, 'height': 480}, 'fps': 30},
            'detection': {'confidence_threshold': 0.25, 'low_confidence_threshold': 0.05},
            'tracking': {'enable': True, 'tracker_type': 'bytetrack.yaml', 'persist_frames': 60},
            'display': {'show_confidence': True, 'show_fps': True, 'fps_smoothing': 0.9},
        }
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        raise

class CameraCalibrator:
    """Handles camera calibration using chessboard pattern"""
    
    def __init__(self, chessboard_size=(9, 6)):
        """
        Initialize the chessboard calibrator.
        
        Args:
            chessboard_size (tuple): Internal corners of the chessboard (width, height)
        """
        self.chessboard_size = chessboard_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints1 = []  # 2d points in camera 1 image plane  
        self.imgpoints2 = []  # 2d points in camera 2 image plane
        
        print(f"📐 Initialized chessboard calibrator with {chessboard_size[0]}x{chessboard_size[1]} pattern")
        
    def find_chessboard_corners(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        """
        Find chessboard corners in both camera images.
        
        Args:
            img1, img2: Camera images
            
        Returns:
            bool: True if corners found in both images
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, self.chessboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, self.chessboard_size, None)
        
        if ret1 and ret2:
            # Refine corners
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), self.criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), self.criteria)
            
            self.objpoints.append(self.objp)
            self.imgpoints1.append(corners1)
            self.imgpoints2.append(corners2)
            
            print(f"✓ Chessboard pattern found in both cameras (total captures: {len(self.objpoints)})")
            return True
        else:
            print(f"⚠ Chessboard pattern not found in both cameras (cam1: {ret1}, cam2: {ret2})")
            return False
    
    def calibrate_homography(self) -> Optional[np.ndarray]:
        """
        Compute homography using RANSAC from collected chessboard corner correspondences.
        
        Returns:
            np.ndarray: Homography matrix or None if calibration fails
        """
        if len(self.imgpoints1) < 7:
            print("❌ Need at least 7 point correspondences for homography")
            return None
            return None
            
        print(f"🔬 Computing homography from {len(self.imgpoints1)} chessboard patterns...")
        
        # Combine all corner points from all frames
        all_points1 = []
        all_points2 = []
        
        for pts1, pts2 in zip(self.imgpoints1, self.imgpoints2):
            all_points1.extend(pts1.reshape(-1, 2))
            all_points2.extend(pts2.reshape(-1, 2))
        
        points1 = np.array(all_points1, dtype=np.float32)
        points2 = np.array(all_points2, dtype=np.float32)
        
        print(f"  Using {len(points1)} point correspondences")
        
        # Compute homography using RANSAC
        # Use simple signature: cv2.findHomography(srcPoints, dstPoints, method, ransacReprojectionThreshold)
        homography, mask = cv2.findHomography(
            points2, points1,  # Transform from camera 2 to camera 1
            cv2.RANSAC,
            5.0    # ransacReprojectionThreshold
        )
        
        if homography is not None:
            inliers = np.sum(mask)
            total_points = len(points1)
            inlier_ratio = inliers / total_points
            
            print("✓ Homography calibration successful")
            print(f"  Inliers: {inliers}/{total_points} ({inlier_ratio:.2%})")
            print(f"  Homography matrix:")
            for row in homography:
                print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}]")
            
            return homography
        else:
            print("❌ Homography calibration failed")
            return None

    def calibrate_stereo(self, img_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Perform stereo calibration using collected chessboard data.
        
        Args:
            img_shape: Image shape (height, width)
            
        Returns:
            dict: Calibration results including rotation, translation, and fundamental matrix
        """
        if len(self.objpoints) < 10:
            print(f"❌ Not enough calibration images: {len(self.objpoints)} < 10")
            return None
            
        print(f"🔬 Performing stereo calibration with {len(self.objpoints)} image pairs...")
        
        # Individual camera calibration
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints1, img_shape[::-1], None, None)
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints2, img_shape[::-1], None, None)
        
        if not ret1 or not ret2:
            print("❌ Individual camera calibration failed")
            return None
            
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints1, self.imgpoints2,
            mtx1, dist1, mtx2, dist2, img_shape[::-1],
            criteria=self.criteria, flags=flags)
        
        if ret:
            print("✓ Stereo calibration successful")
            print(f"  Reprojection error: {ret:.3f}")
            print(f"  Translation: {T.flatten()}")
            print(f"  Rotation angles: {cv2.Rodrigues(R)[0].flatten() * 180 / np.pi}")
            
            return {
                'camera_matrix_1': mtx1,
                'dist_coeffs_1': dist1,
                'camera_matrix_2': mtx2,
                'dist_coeffs_2': dist2,
                'rotation_matrix': R,
                'translation_vector': T,
                'essential_matrix': E,
                'fundamental_matrix': F,
                'reprojection_error': ret
            }
        else:
            print("❌ Stereo calibration failed")
            return None
    
    def triangulate_points(self, points1: np.ndarray, points2: np.ndarray, 
                          calibration_data: Dict) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences.
        
        Args:
            points1, points2: Corresponding 2D points in both cameras
            calibration_data: Stereo calibration results
            
        Returns:
            np.ndarray: 3D points in world coordinates
        """
        # Compute projection matrices
        P1 = np.hstack([calibration_data['camera_matrix_1'], np.zeros((3, 1))])
        
        R = calibration_data['rotation_matrix']
        T = calibration_data['translation_vector']
        RT = np.hstack([R, T])
        P2 = calibration_data['camera_matrix_2'] @ RT
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous coordinates
        
        return points_3d.T
    
    def visualize_chessboard(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Visualize detected chessboard corners"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        ret1, corners1 = cv2.findChessboardCorners(gray1, self.chessboard_size, None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, self.chessboard_size, None)
        
        vis_img1 = img1.copy()
        vis_img2 = img2.copy()
        
        if ret1:
            cv2.drawChessboardCorners(vis_img1, self.chessboard_size, corners1, ret1)
        if ret2:
            cv2.drawChessboardCorners(vis_img2, self.chessboard_size, corners2, ret2)
            
        return np.hstack((vis_img1, vis_img2))

class GlobalTrack:
    """Represents a single global track with all associated data"""
    
    def __init__(self, track_id: int, detection: Dict, current_time: float):
        print(f"GlobalTrack.__init__: Creating track {track_id} for class {detection['class']}")
        
        self.track_id = track_id
        self.position_3d = detection['position_3d']
        self.class_id = detection['class']
        self.confidence = detection['confidence']
        self.created_time = current_time
        self.last_seen = current_time
        self.age = 1
        
        # Store detection data from both cameras
        self.cam1_detection = detection['cam1_detection']
        self.cam2_detection = detection['cam2_detection']
        
        # Store bounding boxes
        self.bbox_cam1 = self._extract_bbox(detection['cam1_detection']) if detection['cam1_detection'] else None
        self.bbox_cam2 = self._extract_bbox(detection['cam2_detection']) if detection['cam2_detection'] else None
        
        # Track type and history
        self.track_type = detection['type']  # 'matched', 'cam1_only', 'cam2_only'
        self.position_history = [self.position_3d]
        self.confidence_history = [self.confidence]
        
    def _extract_bbox(self, detection: List) -> Optional[List[float]]:
        """Extract bounding box from detection"""
        if detection is None:
            return None
        return [float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])]
    
    def update(self, detection: Dict, current_time: float):
        """Update track with new detection data"""
        print(f"GlobalTrack.update: Updating track {self.track_id} with new detection")
        
        self.position_3d = detection['position_3d']
        self.confidence = detection['confidence']
        self.last_seen = current_time
        self.age += 1
        
        # Update detection data
        self.cam1_detection = detection['cam1_detection']
        self.cam2_detection = detection['cam2_detection']
        
        # Update bounding boxes
        self.bbox_cam1 = self._extract_bbox(detection['cam1_detection']) if detection['cam1_detection'] else None
        self.bbox_cam2 = self._extract_bbox(detection['cam2_detection']) if detection['cam2_detection'] else None
        
        # Update track type
        self.track_type = detection['type']
        
        # Add to history (keep last 10 positions)
        self.position_history.append(self.position_3d)
        self.confidence_history.append(self.confidence)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Calculate velocity based on position history"""
        if len(self.position_history) < 2:
            return (0.0, 0.0, 0.0)
        
        # Simple velocity calculation from last two positions
        curr_pos = self.position_history[-1]
        prev_pos = self.position_history[-2]
        
        return (
            curr_pos[0] - prev_pos[0],
            curr_pos[1] - prev_pos[1],
            curr_pos[2] - prev_pos[2]
        )
    
    def get_average_confidence(self) -> float:
        """Get average confidence over history"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
    
    def is_expired(self, current_time: float, timeout: float) -> bool:
        """Check if track has expired"""
        return (current_time - self.last_seen) > timeout
    
    def to_dict(self) -> Dict:
        """Convert track to dictionary for export"""
        return {
            'track_id': self.track_id,
            'position_3d': self.position_3d,
            'class': self.class_id,
            'confidence': self.confidence,
            'created_time': self.created_time,
            'last_seen': self.last_seen,
            'age': self.age,
            'cam1_detection': self.cam1_detection,
            'cam2_detection': self.cam2_detection,
            'bbox_cam1': self.bbox_cam1,
            'bbox_cam2': self.bbox_cam2,
            'track_type': self.track_type,
            'velocity': self.get_velocity(),
            'avg_confidence': self.get_average_confidence()
        }


class MultiCameraYOLOTracker:
    """Main class for multi-camera YOLO tracking system"""
    
    def __init__(self, model_path: str, config: Dict = None):
        # Load configuration
        if config is None:
            config = load_detection_config()
        self.config = config
        
        # Extract camera indices from config (cameras 4 and 6)
        camera_indices = config['cameras']['indices']
        if len(camera_indices) != 2:
            raise ValueError(f"Expected 2 camera indices, got {len(camera_indices)}")
        
        self.camera1_id, self.camera2_id = camera_indices
        print(f"Using cameras: {self.camera1_id}, {self.camera2_id}")
        
        # Set detection parameters from config
        self.confidence_threshold = config['detection']['confidence_threshold']
        print(f"Using confidence threshold: {self.confidence_threshold}")
            
        # Initialize separate YOLO model instances for independent tracking
        try:
            print(f"Loading YOLO model instances from: {model_path}")
            self.model1 = YOLO(model_path)  # For camera 1
            self.model2 = YOLO(model_path)  # For camera 2
            print("✓ Two YOLO model instances loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO models: {e}")
        
        # Initialize cameras with error handling
        self.cap1, self.cap2 = self._initialize_cameras()
        
        # Initialize calibrator with camera model
        self.calibrator = CameraCalibrator()
        self.homography = None  
        self.calibration_data = None
        self.calibrated = False

        
        # Get target resolution from config
        resolution = config['cameras']['resolution']
        self.target_resolution = (resolution['width'], resolution['height'])
        
        # Global tracking for 3D coordinates
        self.global_tracks = {}
        self.next_global_track_id = 1
        
        # Timing control for printing
        self.last_print_time = 0
        self.print_interval = 2.0  # Print every 2 seconds
        
        # Initialize trackers (will be set up after calibration)
        self.global_tracker = None
    
    def _initialize_cameras(self) -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
        """Initialize cameras with proper error handling"""
        print(f"Initializing cameras: {self.camera1_id}, {self.camera2_id}")
        
        cap1 = cv2.VideoCapture(self.camera1_id)
        cap2 = cv2.VideoCapture(self.camera2_id)
        
        # Check if cameras opened successfully
        if not cap1.isOpened():
            if cap2.isOpened():
                cap2.release()
            raise RuntimeError(f"Failed to open camera {self.camera1_id}")
            
        if not cap2.isOpened():
            cap1.release()
            raise RuntimeError(f"Failed to open camera {self.camera2_id}")
        
        # Set camera properties from config
        cameras = [(cap1, self.camera1_id), (cap2, self.camera2_id)]
        cam_config = self.config['cameras']
        target_width = cam_config['resolution']['width']
        target_height = cam_config['resolution']['height']
        target_fps = cam_config['fps']
        
        for cap, cam_id in cameras:
            try:
                # Set resolution and FPS
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                
                # Set buffer size and format if specified  
                if 'buffer_size' in cam_config:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, cam_config['buffer_size'])
                if cam_config.get('format') == 'MJPG':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                elif cam_config.get('format') == 'YUYV':
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                
                # Set camera to automatic adjustments
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Auto exposure mode (3 = full auto)
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)        # Auto white balance
                
                # Let camera 6 use automatic settings (don't override defaults)
                if cam_id == self.camera2_id:  # Camera 6 
                    print(f"  Using automatic settings for camera {cam_id}")
                    # Don't set manual values - let camera auto-adjust
                
                # Verify settings
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✓ Camera {cam_id}: {int(width)}x{int(height)} @ {fps:.1f}fps")
                
                # Test frame capture
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Camera {cam_id} cannot capture frames")
                
                # If frame size doesn't match target, we'll resize in the main loop
                actual_height, actual_width = frame.shape[:2]
                if actual_width != target_width or actual_height != target_height:
                    print(f"  Note: Camera {cam_id} will be resized from {actual_width}x{actual_height} to {target_width}x{target_height}")
                    
            except Exception as e:
                cap1.release()
                cap2.release()
                raise RuntimeError(f"Failed to configure camera {cam_id}: {e}")
        
        print("✓ Cameras initialized successfully")
        self.target_resolution = (target_width, target_height)
        return cap1, cap2
        
    def calibrate_cameras(self) -> bool:
        """Calibrate cameras using selected hemography"""
        print("Hold a chessboard pattern visible in both cameras")
        print("Press 'c' to capture calibration frame, 'q' to finish calibration")
        min_frames = 7
        print(f"Need at least {min_frames} good chessboard captures for calibration")
        
        self.calibrated = False  # Ensure calibration flag is reset

        while True:
            # Read frames
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print("Failed to read from cameras")
                return False
            
            # Show chessboard visualization
            vis_frame = self.calibrator.visualize_chessboard(frame1, frame2)
            
            # Add text overlay
            cv2.putText(vis_frame, f"Calibration - Frames: {len(self.calibrator.objpoints)}/{min_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, "Press 'c' to capture, 'q' to finish", 
                       (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Chessboard Calibration - Camera 1 | Camera 2', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print("Capturing chessboard frame...")
                if self.calibrator.find_chessboard_corners(frame1, frame2):
                    if len(self.calibrator.objpoints) >= min_frames:
                        print(f"Sufficient calibration frames captured. Press 'q' to perform calibration.")
                else:
                    print("Chessboard not found in both cameras. Try adjusting position.")
                    
            elif key == ord('q'):
                if len(self.calibrator.objpoints) < min_frames:
                    print(f"Not enough calibration frames ({len(self.calibrator.objpoints)}/{min_frames}). Calibration aborted.")
                    cv2.destroyAllWindows()
                    return False
                print("Performing homography calibration...")
                self.homography = self.calibrator.calibrate_homography()
                if self.homography is not None:
                    self.calibrated = True
                    print("Calibration completed successfully.")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Calibration failed. Please try again.")
                    cv2.destroyAll
                    return False

        cv2.destroyAllWindows()
        return False
    
    def detect_objects(self, frame: np.ndarray, camera_id: int) -> List:
        """Detect objects using YOLO model with ByteTrack tracking"""
        print(f"MultiCameraYOLOTracker.detect_objects: Starting detection for camera {camera_id}")
        
        if frame is None or frame.size == 0:
            print("MultiCameraYOLOTracker.detect_objects: Empty detection frame received")
            return []
        
        # Use appropriate model instance based on camera
        model = self.model1 if camera_id == 1 else self.model2
        
        # Use YOLO with ByteTrack tracking enabled
        results = model.track(
            frame, 
            conf=self.confidence_threshold, 
            tracker='bytetrack.yaml',
            verbose=False,
            persist=True  # Keep track IDs consistent across frames
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                    
                    detections.append([x1, y1, w, h, conf, cls, track_id])
        
        print(f"MultiCameraYOLOTracker.detect_objects: Camera {camera_id} produced {len(detections)} detections")
        return detections

    def calculate_3d_coordinates(self, detections_cam1: List, detections_cam2: List) -> None:
        """Update global tracks from global tracker and calculate 3D coordinates"""
        print(f"MultiCameraYOLOTracker.calculate_3d_coordinates: Starting with calibrated={self.calibrated}")
        
        if not self.calibrated:
            return
        
        # The global tracker now handles all the correlation and track management
        # Just extract the current global tracks for display
        current_time = time.time()
        self.global_tracks.clear()  # Clear old tracks
        
        if hasattr(self, 'tracker') and self.tracker:
            for track_id, track in self.tracker.tracks.items():
                # Get class name
                class_id = int(track.class_id)
                class_name = self.model1.names[class_id] if class_id < len(self.model1.names) else f"class_{class_id}"
                
                # Create global track entry using track object data
                self.global_tracks[track_id] = {
                    'coordinates_3d': track.position_3d,
                    'class_name': class_name,
                    'confidence': track.confidence,
                    'track_id_cam1': track.cam1_detection[6] if track.cam1_detection and len(track.cam1_detection) > 6 else None,
                    'track_id_cam2': track.cam2_detection[6] if track.cam2_detection and len(track.cam2_detection) > 6 else None,
                    'last_seen': track.last_seen,
                    'age': track.age,
                    'center_cam1': track.position_3d[:2],  # Use 3D position for 2D center
                    'center_cam2': None,  # Could calculate from cam2_detection if needed
                    'bbox_cam1': track.bbox_cam1,
                    'bbox_cam2': track.bbox_cam2,
                    'velocity': track.get_velocity(),
                    'avg_confidence': track.get_average_confidence(),
                    'track_type': track.track_type
                }
        
        print(f"MultiCameraYOLOTracker.calculate_3d_coordinates: Updated {len(self.global_tracks)} global tracks")

    def draw_detections(self, frame: np.ndarray, detections: List, 
                       color: Tuple[int, int, int], prefix: str, track_ids: Dict = None, model=None, border_color=(0,0,0)) -> np.ndarray:
        """Draw a colored square and/or circle at the center of each bounding box depending on class, with class color and black border"""
        print(f"MultiCameraYOLOTracker.draw_detections: Drawing {len(detections)} detections with prefix {prefix}")
        
        frame_copy = frame.copy()
        
        # Use model1 as default if no model specified
        if model is None:
            model = self.model1
        
        for i, det in enumerate(detections):
            if len(det) >= 7:  # Has track_id
                x, y, w, h, conf, cls, track_id = det
            else:  # Original format
                x, y, w, h, conf, cls = det[:6]
                track_id = None
            
            # Convert to scalars explicitly
            x, y, w, h = float(x), float(y), float(w), float(h)
            conf = float(conf)
            cls = int(cls)

            # Set color based on class
            if cls in [0, 1]:
                base_color = (0, 0, 255)   # Red (BGR)
            elif cls in [2, 3]:
                base_color = (0, 255, 0)   # Green
            elif cls in [4, 5]:
                base_color = (255, 0, 0)   # Blue
            else:
                base_color = color         # fallback to input color

            # Adjust color based on confidence level
            if conf > 0.7:
                box_color = base_color
            elif conf > 0.5:
                box_color = tuple([int(c * 0.6) for c in base_color])
            else:
                box_color = tuple([int(c * 0.3) for c in base_color])
            
            center_x, center_y = int(x + w/2), int(y + h/2)
            # Draw square for classes 1, 3, 5
            if cls in [0, 2, 4]:
                # Draw black border (thicker)
                cv2.rectangle(frame_copy, (center_x-8, center_y-8), (center_x+8, center_y+8), border_color, -1)
                # Draw filled color square (smaller)
                cv2.rectangle(frame_copy, (center_x-5, center_y-5), (center_x+5, center_y+5), base_color, -1)
            # Draw circle for classes 2, 4, 5
            if cls in [1, 3, 5]:
                # Draw black border (thicker)
                cv2.circle(frame_copy, (center_x, center_y), 10, border_color, -1)
                # Draw filled color circle (smaller)
                cv2.circle(frame_copy, (center_x, center_y), 7, base_color, -1)
        
        return frame_copy

    def transform_detections_to_cam1(self, detections_cam2: List) -> List:
        """Transform camera 2 detections to camera 1 coordinate system"""
        print(f"MultiCameraYOLOTracker.transform_detections_to_cam1: Transforming {len(detections_cam2)} detections")
        
        if self.homography is None:
            print("MultiCameraYOLOTracker.transform_detections_to_cam1: No homography available")
            return []
            
        transformed_detections = []
        
        for det in detections_cam2:
            if len(det) < 6:
                continue
                
            x, y, w, h, conf, cls = det[:6]
            
            # Convert to scalars
            x, y, w, h, conf = float(x), float(y), float(w), float(h), float(conf)
            cls = int(cls)
            
            # Transform corners of bounding box
            corners = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x, y + h]],
                [[x + w, y + h]]
            ], dtype=np.float32)
            
            transformed_corners = cv2.perspectiveTransform(corners, self.homography)
            
            # Get new bounding box
            x_coords = transformed_corners[:, 0, 0]
            y_coords = transformed_corners[:, 0, 1]
            
            new_x = max(0, int(np.min(x_coords)))
            new_y = max(0, int(np.min(y_coords)))
            new_w = int(np.max(x_coords) - np.min(x_coords))
            new_h = int(np.max(y_coords) - np.min(y_coords))
            
            if new_w > 0 and new_h > 0:  # Valid bounding box
                transformed_detections.append([new_x, new_y, new_w, new_h, conf, cls])
        
        print(f"MultiCameraYOLOTracker.transform_detections_to_cam1: Transformed {len(transformed_detections)} valid detections")
        return transformed_detections

    def run(self):
        """Main tracking loop"""
        print("MultiCameraYOLOTracker.run: Starting main tracking loop")
        
        if not self.calibrated:
            print("MultiCameraYOLOTracker.run: Not calibrated, starting calibration")
            if not self.calibrate_cameras():
                print("MultiCameraYOLOTracker.run: Camera calibration failed")
                return
        
        print("MultiCameraYOLOTracker.run: Starting multi-camera tracking...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Recalibrate cameras")
        print("  'd' - Toggle dual view mode")
        print("  'i' - Toggle track info display")
        print("  't' - Reset all tracks (use if too many tracks)")
        
        dual_view = False
        show_info = True

        # Set desired GUI scale factor (e.g., 1.5x larger)
        gui_scale = 2

        while True:
            # Read frames
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()
            
            if not ret1 or not ret2:
                print("MultiCameraYOLOTracker.run: Failed to read from cameras")
                break
            
            # Detect objects in both cameras with separate model instances
            detections_cam1 = self.detect_objects(frame1, camera_id=1)
            detections_cam2 = self.detect_objects(frame2, camera_id=2)
            
            ######################################################################################3#############
            # MATCHING ALGORITHM, INSERT HERE
            #####################################################################################################

            # Update tracker with detections
            #if hasattr(self, 'tracker') and self.tracker:
            #    self.tracker.update_tracks(detections_cam1, detections_cam2)
            
            # Calculate 3D coordinates from corresponding detections
            #self.calculate_3d_coordinates(detections_cam1, detections_cam2)
            
            # Print detection details every 2 seconds
            #current_time = time.time()
            #if current_time - self.last_print_time >= self.print_interval:
            #    self.print_detection_summary(detections_cam1, detections_cam2)
            #    self.last_print_time = current_time
            
            if dual_view:
                # Show both camera views side by side with ALL detections (no filtering)
                display_frame1 = self.draw_detections(frame1, detections_cam1, 
                                                    (0, 255, 0), "Cam1", {}, self.model1,(128, 0, 128))  # Green for Camera 1
                display_frame2 = self.draw_detections(frame2, detections_cam2, 
                                                    (255, 0, 0), "Cam2", {}, self.model2,(128,128,0))  # Blue for Camera 2
                
                # Add detailed detection info to each frame
                cv2.putText(display_frame1, f"Camera 1 - Raw Detections: {len(detections_cam1)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame2, f"Camera 2 - Raw Detections: {len(detections_cam2)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Add frame labels
                cv2.putText(display_frame1, "CAMERA 1", (10, display_frame1.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame2, "CAMERA 2", (10, display_frame2.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                
                # Resize frames if needed to fit side by side
                h1, w1 = display_frame1.shape[:2]
                h2, w2 = display_frame2.shape[:2]
                max_height = max(h1, h2)
                
                if h1 != max_height:
                    display_frame1 = cv2.resize(display_frame1, (int(w1 * max_height / h1), max_height))
                if h2 != max_height:
                    display_frame2 = cv2.resize(display_frame2, (int(w2 * max_height / h2), max_height))
                
                display_frame = np.hstack((display_frame1, display_frame2))
                window_title = 'Multi-Camera Tracking - Dual View (Raw Detections)'
            else:
                # Show only camera 1 with transformed detections from camera 2
                transformed_cam2_dets = self.transform_detections_to_cam1(detections_cam2)
                
                # Draw detections on camera 1 frame
                display_frame = self.draw_detections(frame1, detections_cam1, 
                                                   (0, 255, 0), "Cam1", {}, self.model1,(128, 0, 128))  # Green for Camera 1
                display_frame = self.draw_detections(display_frame, transformed_cam2_dets, 
                                                   (0, 0, 255), "Cam2", {}, self.model2,(128, 128, 0))  # Yellow for Camera 2
                window_title = 'Multi-Camera Tracking - Overlay View'
            
            # Add information overlay
            if show_info:
                self._draw_info_overlay(display_frame)
            
            # --- Resize GUI window larger ---
            display_frame = cv2.resize(
                display_frame,
                (int(display_frame.shape[1] * gui_scale), int(display_frame.shape[0] * gui_scale)),
                interpolation=cv2.INTER_LINEAR
            )
            # --------------------------------

            # Display
            cv2.imshow(window_title, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("MultiCameraYOLOTracker.run: Recalibrating cameras...")
                cv2.destroyAllWindows()
                self.calibrated = False
                self.homography = None
                self.tracker = None
                if not self.calibrate_cameras():
                    print("MultiCameraYOLOTracker.run: Recalibration failed")
                    break
            elif key == ord('d'):
                dual_view = not dual_view
                cv2.destroyAllWindows()
                print(f"MultiCameraYOLOTracker.run: Dual view mode: {'ON' if dual_view else 'OFF'}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"MultiCameraYOLOTracker.run: Info display: {'ON' if show_info else 'OFF'}")
            elif key == ord('t'):
                # Reset tracker
                if hasattr(self, 'tracker') and self.tracker:
                    self.tracker.tracks.clear()
                    self.tracker.next_track_id = 0
                    print("MultiCameraYOLOTracker.run: Global tracker manually reset")
        
        print("MultiCameraYOLOTracker.run: Exiting main loop, cleaning up")
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'cap1') and self.cap1.isOpened():
                self.cap1.release()
                print("✓ Camera 1 released")
        except Exception as e:
            print(f"Warning: Error releasing camera 1: {e}")
            
        try:
            if hasattr(self, 'cap2') and self.cap2.isOpened():
                self.cap2.release()
                print("✓ Camera 2 released")
        except Exception as e:
            print(f"Warning: Error releasing camera 2: {e}")
            
        try:
            cv2.destroyAllWindows()
            print("✓ Windows closed")
        except Exception as e:
            print(f"Warning: Error destroying windows: {e}")
    
    def _draw_info_overlay(self, frame):
        # Draws a simple info overlay in the top-left corner
        text = "Press 'q' to quit | 'r' to recalibrate | 'd' dual view | 'i' info | 't' reset tracks"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

def main():
    print("Multi-Camera YOLO Tracker with 3D Coordinates")
    print("=" * 50)
    
    # Load configuration
    print("⚙️ Loading configuration...")
    config = load_detection_config()
    
    # Auto-detect the latest model
    print("🔍 Auto-detecting latest trained model...")
    model_path, model_name = find_latest_model()
    
    if model_path is None:
        print("❌ No trained models found!")
        print("Please train a model first using train_yolo_parts.py")
        return
    
    print(f"✓ Using model: {model_name}")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Cameras: {config['cameras']['indices']}")
    print(f"✓ Confidence threshold: {config['detection']['confidence_threshold']}")
    print(f"✓ Tracking: {config['tracking']['tracker_type']}")
    print()
    
    try:
        # Initialize tracker with configuration
        print("Initializing Multi-Camera YOLO Tracker...")
        tracker = MultiCameraYOLOTracker(model_path, config=config)
        
        # Run the tracker
        tracker.run()
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("Please check that the model path and camera IDs are correct.")
    except RuntimeError as e:
        print(f"✗ Runtime error: {e}")
        print("Please check camera connections and permissions.")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        try:
            if 'tracker' in locals():
                tracker.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()