from .ModelPlus import ModelPlus
from .MCMOTUtils import MCMOTUtils
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from .config.ArucoConfig import ArucoConfig

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
detector = cv2.aruco.ArucoDetector(aruco_dict)

class Camera:    
    def __init__(
        self,
        camera_number,
        camera_device_id,
        camera_name,
        model_path=None,
        confidence_threshold=None,
        tracker_yaml_path=None,
        aruco_positions=None,
        display=True,
        calibration_method=None,
        calibration=None,
        load_intrinsics=True,
        show_world_axis=True,
        world_axis_length=3.0,
    ):
        if calibration is not None:
            calibration_method = calibration

        self.camera_number = camera_number
        self.camera_device_id = camera_device_id
        print(f"    Opening Camera on Port {camera_device_id}")
        self.cap = cv2.VideoCapture(camera_device_id)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device id {camera_device_id} ({camera_name}). "
                "Check camera permissions and device index."
            )
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
        self.is_ground_referenced = False
        self.world_axis_c = None
        self.fov_ground_pts = None
        self.calibration_method = calibration_method
        self.display_window_name = None
        self.gui_available = MCMOTUtils.has_highgui()
        self.rvec = None
        self.tvec = None
        self.Rt = None
        self.projMatrix = None
        self.r = None
        self.t = None
        self.intrinsics_loaded = False
        self.show_world_axis = bool(show_world_axis)
        self.world_axis_length = float(world_axis_length)
        self._warned_arrow_draw_failure = False

        if load_intrinsics:
            self.set_intrinsics()

        self.model_plus = None
        if model_path is not None:
            self.attach_model(model_path, confidence_threshold, tracker_yaml_path)

        if self.calibration_method is None and aruco_positions is not None:
            self.calibration_method = "aruco"

        if self.calibration_method is not None:
            self.calibration_method = self.calibration_method.lower()
            if self.calibration_method not in {"aruco", "landmarks", "auto"}:
                raise ValueError(
                    f"Unknown calibration_method '{calibration_method}'. "
                    "Expected 'aruco', 'landmarks', or 'auto'."
                )

            self.aruco_positions = aruco_positions
            if self.calibration_method in {"aruco", "landmarks"}:
                self.world_calibration_status = self.set_extrinsics()
            elif self.calibration_method == "auto":
                print(
                    f"    Auto calibration for {self.camera_name} will run after all cameras are initialized."
                )
        else:
            self.aruco_positions = None

        if display:
            if self.gui_available:
                self.display_window_name = f"Camera {self.camera_name}"
                cv2.namedWindow(self.display_window_name, cv2.WINDOW_NORMAL)
            else:
                print(
                    f"Camera {self.camera_name}: OpenCV HighGUI unavailable; "
                    "camera display window disabled."
                )

    def set_intrinsics(self):
        package_root = Path(__file__).resolve().parent
        rel_path = Path("config") / "camera_calibration" / self.camera_name / f"{self.camera_name}.npz"
        candidate_paths = [
            Path.cwd() / rel_path,         # keep backward compatibility for old run locations
            package_root / rel_path,       # robust package-relative lookup
        ]

        calibration_path = next((p for p in candidate_paths if p.exists()), None)
        if calibration_path is None:
            calib_root = package_root / "config" / "camera_calibration"
            available = []
            if calib_root.exists():
                available = sorted([p.name for p in calib_root.iterdir() if p.is_dir()])
            raise FileNotFoundError(
                "Camera calibration file not found.\n"
                f"Camera name: '{self.camera_name}'\n"
                f"Tried: {[str(p) for p in candidate_paths]}\n"
                f"Available calibration folders: {available}"
            )

        calibration_data = np.load(str(calibration_path))
        self.mtx = calibration_data['mtx']
        self.dist = calibration_data['dist']
        self.intrinsics_loaded = True

    def attach_model(self, model_path, confidence_threshold=0.1, tracker_yaml_path="./config/bytetrack.yaml"):
        self.model_plus = ModelPlus(model_path, confidence_threshold, tracker_yaml_path)
        return self.model_plus

    def detach_model(self):
        self.model_plus = None

    def set_extrinsics(self, reference_camera=None, auto_baseline=1.0):
        if self.calibration_method in {"aruco", "landmarks"} and not self.gui_available:
            raise RuntimeError(
                "Interactive camera calibration requires OpenCV HighGUI window support, "
                "but this OpenCV build has no GUI backend. "
                "Install GUI-enabled OpenCV dependencies (GTK/Qt/Cocoa) or use precomputed extrinsics."
            )

        if self.calibration_method == "aruco":
            self.aruco_config = ArucoConfig("square4")
            return self.cal_aruco()
        if self.calibration_method == "landmarks":
            return self.cal_landmarks()
        if self.calibration_method == "auto":
            return self.cal_auto(reference_camera=reference_camera, baseline=auto_baseline)
        raise ValueError(
            f"Unknown calibration method '{self.calibration_method}'. "
            "Expected 'aruco', 'landmarks', or 'auto'."
        )

    def set_extrinsics_from_rt(self, R, tvec, world_calibrated=True):
        self.r = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        self.t = self.tvec
        self.rvec, _ = cv2.Rodrigues(self.r)
        self.Rt = np.hstack((self.r, self.tvec))
        self.projMatrix = np.dot(self.mtx, self.Rt)
        self.world_axis_c = None
        self.fov_ground_pts = None
        self.world_calibration_status = bool(world_calibrated)
        self.is_ground_referenced = bool(world_calibrated)
        if world_calibrated:
            self.compute_fov_on_ground()
        return True

    def cal_aruco(self):
        # Show Camera To Make Sure ArUco Markers are Visible
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.frame_dims is None:
                self.frame_dims = np.shape(gray)
            corners, ids, rejected = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    c = corner[0]
                    center = self._to_cv_point(c.mean(axis=0))
                    if center is not None:
                        cv2.putText(
                            frame,
                            f"ID:{ids[i][0]}",
                            center,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    pt1 = self._to_cv_point(c[0])
                    pt2 = self._to_cv_point(c[1])
                    if pt1 is not None and pt2 is not None:
                        self._draw_arrow(frame, pt1, pt2, (255, 0, 0), thickness=2, tip_length=0.3)
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
            solved, rvec, tvec = cv2.solvePnP(object_points, image_points, self.mtx, self.dist)
            if not solved:
                print(f"Error: solvePnP failed for camera {self.camera_name}.")
                return False
            self.set_extrinsics_from_rt(cv2.Rodrigues(rvec)[0], tvec, world_calibrated=True)
            print(f"Camera {self.camera_name} extrinsics set.")
        else:
            print(f"Error: Not enough ArUco markers detected for camera {self.camera_name}. Need at least 4.")
            return False
        return True

    def _prompt_landmark_coordinate(self, pixel_xy):
        while True:
            raw = input(
                f"Enter world coordinate for pixel {pixel_xy} as x,y,z "
                "(or type 'skip' to discard): "
            ).strip()
            if raw.lower() == "skip":
                return None

            parts = [p for p in raw.replace(",", " ").split() if p]
            if len(parts) != 3:
                print("Expected exactly 3 numeric values: x y z")
                continue

            try:
                return [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                print("Invalid numeric input. Try again.")

    def cal_landmarks(self):
        window_name = f"Camera {self.camera_name} - Landmark Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"[{self.camera_name}] Landmark calibration controls:")
        print("  1) Left click a landmark in the image")
        print("  2) Press SPACE or ENTER to add the clicked point")
        print("  3) Enter world coordinate as x,y,z in the terminal")
        print("  4) Press 'c' to complete calibration (requires at least 4 points)")

        selected = {"pixel": None}
        image_points = []
        object_points = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected["pixel"] = (x, y)

        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if self.frame_dims is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_dims = np.shape(gray)

            preview = frame.copy()

            for idx, pt in enumerate(image_points):
                px = (int(pt[0]), int(pt[1]))
                cv2.circle(preview, px, 5, (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"P{idx + 1}",
                    (px[0] + 8, px[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            if selected["pixel"] is not None:
                cv2.circle(preview, selected["pixel"], 6, (0, 0, 255), -1)

            cv2.putText(
                preview,
                "Click landmark, SPACE/ENTER add, c complete",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                preview,
                f"Points: {len(image_points)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("c"), ord("C")):
                if len(image_points) < 4:
                    print("Need at least 4 landmarks before completing calibration.")
                    continue
                break

            if key in (13, 10, 32):  # Enter, Linefeed, Space
                if selected["pixel"] is None:
                    print("No landmark selected. Click a point first.")
                    continue

                world_point = self._prompt_landmark_coordinate(selected["pixel"])
                if world_point is None:
                    selected["pixel"] = None
                    continue

                image_points.append([float(selected["pixel"][0]), float(selected["pixel"][1])])
                object_points.append(world_point)
                print(
                    f"Added landmark {len(image_points)}: "
                    f"pixel={selected['pixel']} world={world_point}"
                )
                selected["pixel"] = None

        cv2.destroyWindow(window_name)

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        solved, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.mtx,
            self.dist,
            reprojectionError=4.0,
            iterationsCount=1000,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            print(f"Error: solvePnPRansac failed for camera {self.camera_name}.")
            return False
        if inliers is None or len(inliers) < 4:
            print(f"Error: solvePnPRansac produced too few inliers for camera {self.camera_name}.")
            return False

        self.set_extrinsics_from_rt(cv2.Rodrigues(rvec)[0], tvec, world_calibrated=True)
        print(f"Camera {self.camera_name} landmark extrinsics set.")
        return True

    def cal_auto(self, reference_camera, baseline=1.0, max_trials=60, min_inliers=8):
        if reference_camera is None:
            raise ValueError("Auto calibration requires a reference_camera.")
        if reference_camera is self:
            raise ValueError("Auto calibration reference_camera must be a different camera.")
        if reference_camera.cap is None or self.cap is None:
            print(f"Error: auto calibration failed for camera {self.camera_name} (camera handle missing).")
            return False
        if not reference_camera.cap.isOpened() or not self.cap.isOpened():
            print(
                f"Error: auto calibration failed for camera {self.camera_name} "
                f"(camera stream not open, ref_open={reference_camera.cap.isOpened()}, "
                f"cur_open={self.cap.isOpened()})."
            )
            return False

        print(
            f"Auto-calibrating {self.camera_name} against {reference_camera.camera_name} "
            "using feature matching + RANSAC."
        )

        warmup_frames = 30
        for _ in range(warmup_frames):
            reference_camera.cap.read()
            self.cap.read()
            time.sleep(0.01)

        detectors = [("ORB", cv2.ORB_create(nfeatures=7000, fastThreshold=6))]
        if hasattr(cv2, "AKAZE_create"):
            detectors.append(("AKAZE", cv2.AKAZE_create()))
        if hasattr(cv2, "SIFT_create"):
            detectors.append(("SIFT", cv2.SIFT_create(nfeatures=3000)))

        best_pose = None
        best_debug = {"detector": None, "matches": 0, "inliers": 0, "parallax": 0.0}
        stats = {
            "read_fail": 0,
            "low_texture": 0,
            "descriptor_fail": 0,
            "few_keypoints": 0,
            "few_matches": 0,
            "essential_fail": 0,
            "pose_fail": 0,
        }
        last_ref_frame = None
        last_cur_frame = None

        for _ in range(max_trials):
            ok_ref, frame_ref = reference_camera.cap.read()
            ok_cur, frame_cur = self.cap.read()
            if not (ok_ref and ok_cur) or frame_ref is None or frame_cur is None:
                stats["read_fail"] += 1
                continue

            last_ref_frame = frame_ref.copy()
            last_cur_frame = frame_cur.copy()

            if reference_camera.frame_dims is None:
                reference_camera.frame_dims = frame_ref.shape[:2]
            if self.frame_dims is None:
                self.frame_dims = frame_cur.shape[:2]

            gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
            gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)

            std_ref = float(np.std(gray_ref))
            std_cur = float(np.std(gray_cur))
            if min(std_ref, std_cur) < 6.0:
                stats["low_texture"] += 1
                continue

            gray_ref = cv2.equalizeHist(gray_ref)
            gray_cur = cv2.equalizeHist(gray_cur)

            for detector_name, detector in detectors:
                kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
                kp_cur, des_cur = detector.detectAndCompute(gray_cur, None)

                if des_ref is None or des_cur is None:
                    stats["descriptor_fail"] += 1
                    continue
                if len(kp_ref) < 16 or len(kp_cur) < 16:
                    stats["few_keypoints"] += 1
                    continue

                if detector_name == "SIFT":
                    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                    norm = cv2.NORM_L2
                    ratio = 0.8
                else:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    norm = cv2.NORM_HAMMING
                    ratio = 0.85

                matches = matcher.knnMatch(des_ref, des_cur, k=2)
                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < ratio * n.distance:
                        good.append(m)

                if len(good) < 12:
                    matcher_cc = cv2.BFMatcher(norm, crossCheck=True)
                    cc_matches = matcher_cc.match(des_ref, des_cur)
                    cc_matches = sorted(cc_matches, key=lambda m: m.distance)
                    if len(cc_matches) >= 12:
                        good = cc_matches[: min(800, len(cc_matches))]

                if len(good) < 12:
                    stats["few_matches"] += 1
                    if len(good) > best_debug["matches"]:
                        best_debug["matches"] = len(good)
                        best_debug["detector"] = detector_name
                    continue

                pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts_cur = np.float32([kp_cur[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                pts_ref_n = cv2.undistortPoints(pts_ref, reference_camera.mtx, reference_camera.dist)
                pts_cur_n = cv2.undistortPoints(pts_cur, self.mtx, self.dist)

                for ransac_threshold in (0.003, 0.006, 0.01):
                    E, mask_E = cv2.findEssentialMat(
                        pts_ref_n,
                        pts_cur_n,
                        focal=1.0,
                        pp=(0.0, 0.0),
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=ransac_threshold,
                    )
                    if E is None:
                        stats["essential_fail"] += 1
                        continue

                    if E.shape == (3, 3):
                        candidates = [E]
                    else:
                        candidates = [
                            E[i:i + 3, :]
                            for i in range(0, E.shape[0], 3)
                            if i + 3 <= E.shape[0]
                        ]

                    for E_i in candidates:
                        try:
                            inliers, R, t, pose_mask = cv2.recoverPose(
                                E_i, pts_ref_n, pts_cur_n, mask=mask_E
                            )
                        except cv2.error:
                            stats["pose_fail"] += 1
                            continue

                        inliers = int(inliers)
                        if inliers <= 0:
                            stats["pose_fail"] += 1
                            continue

                        if pose_mask is not None:
                            pm = pose_mask.ravel().astype(bool)
                            if np.any(pm):
                                parallax = float(np.median(np.linalg.norm(
                                    pts_ref_n[pm].reshape(-1, 2) - pts_cur_n[pm].reshape(-1, 2),
                                    axis=1
                                )))
                            else:
                                parallax = 0.0
                        else:
                            parallax = 0.0

                        if inliers > best_debug["inliers"] or (
                            inliers == best_debug["inliers"] and parallax > best_debug["parallax"]
                        ):
                            best_debug = {
                                "detector": detector_name,
                                "matches": len(good),
                                "inliers": inliers,
                                "parallax": parallax,
                            }
                            best_pose = (R, t)

        if best_pose is None or best_debug["inliers"] < min_inliers:
            debug_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
            debug_ref_path = None
            debug_cur_path = None
            if last_ref_frame is not None and last_cur_frame is not None:
                try:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    stamp = int(time.time())
                    debug_ref_path = debug_dir / f"auto_calib_ref_cam{reference_camera.camera_number}_{stamp}.jpg"
                    debug_cur_path = debug_dir / f"auto_calib_cam{self.camera_number}_{stamp}.jpg"
                    cv2.imwrite(str(debug_ref_path), last_ref_frame)
                    cv2.imwrite(str(debug_cur_path), last_cur_frame)
                except Exception:
                    debug_ref_path = None
                    debug_cur_path = None

            print(
                f"Error: auto calibration failed for camera {self.camera_name}. "
                f"Best result detector={best_debug['detector']}, matches={best_debug['matches']}, "
                f"inliers={best_debug['inliers']}, parallax={best_debug['parallax']:.4f}. "
                f"Stats={stats}."
            )
            if debug_ref_path and debug_cur_path:
                print(f"Saved debug frames: ref={debug_ref_path}, cam={debug_cur_path}")
            print("Ensure both cameras have overlapping view, good lighting, and textured landmarks.")
            return False

        R, t = best_pose
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-9:
            print(f"Error: auto calibration failed for camera {self.camera_name} (degenerate translation).")
            return False

        scale = float(baseline) if baseline and baseline > 0 else 1.0
        t = (t / norm_t) * scale
        self.set_extrinsics_from_rt(R, t, world_calibrated=False)
        print(
            f"Camera {self.camera_name} auto extrinsics set "
            f"(detector={best_debug['detector']}, matches={best_debug['matches']}, "
            f"inliers={best_debug['inliers']}, baseline={scale})."
        )
        return True

    def capture_frame(self):
        ret, self.frame = self.cap.read()
        if ret:
            if self.frame_dims is None and self.frame is not None:
                # Store as (height, width) to match numpy/OpenCV shape convention.
                self.frame_dims = self.frame.shape[:2]
            self.frame_time = time.time()
            self.frame_annotated = self.frame.copy()


    def detect_and_track(self):
        self.fps_counter.tick()
        if self.frame_time is not None and (time.time() - self.frame_time) > 0.25:
            print("WARNING: Running detect_and_track() on a frame older than 0.25s")
        if self.model_plus is None or self.frame is None:
            return
        self.model_plus.detect_and_track(self.frame)

    def annotate_frame(self):
        if self.model_plus is not None:
            self.model_plus.annotate_frame()
            af = (
                self.model_plus.frame_annotated.copy()
                if self.model_plus.frame_annotated is not None
                else None
            )
        else:
            af = self.frame.copy() if self.frame is not None else None

        if af is None:
            self.frame_annotated = None
            return None

        if self.show_world_axis and self._can_draw_world_axis():
            af = self.draw_world_axis(af)
        
        fps_text = f"FPS: {self.fps_counter.get_fps():.1f}"
        af = cv2.putText(af, fps_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0,225,0), 2)

        self.frame_annotated = af
        return af
    
    def display_frame(self):
        if self.display_window_name and self.gui_available and self.frame_annotated is not None:
            cv2.imshow(self.display_window_name, self.frame_annotated)
            cv2.waitKey(1)

    @staticmethod
    def _to_cv_point(point_like):
        int_limit = 2_147_483_000  # conservative bound for 32-bit OpenCV point ints
        try:
            x = float(point_like[0])
            y = float(point_like[1])
        except Exception:
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        if abs(x) > int_limit or abs(y) > int_limit:
            return None
        return (int(round(x)), int(round(y)))

    def _draw_arrow(self, frame, pt1, pt2, color, thickness=2, tip_length=0.2):
        p1 = self._to_cv_point(pt1)
        p2 = self._to_cv_point(pt2)
        if p1 is None or p2 is None:
            return False

        try:
            cv2.arrowedLine(
                frame,
                p1,
                p2,
                color,
                int(thickness),
                tipLength=float(tip_length),
            )
            return True
        except Exception as exc:
            try:
                cv2.line(frame, p1, p2, color, int(thickness))
            except Exception:
                pass
            if not self._warned_arrow_draw_failure:
                print(
                    "WARNING: cv2.arrowedLine failed once; falling back to cv2.line. "
                    f"pt1={p1} ({type(p1[0]).__name__}), "
                    f"pt2={p2} ({type(p2[0]).__name__}), error={exc}"
                )
                self._warned_arrow_draw_failure = True
            return False

    def _can_draw_world_axis(self):
        required = (self.rvec, self.tvec, self.mtx, self.dist)
        return all(value is not None for value in required)

    def draw_world_axis(self, frame):
        if self.world_axis_c is None:
            axis_length = max(0.1, float(self.world_axis_length))
            axis_points_w = np.float32(
                [
                    [0.0, 0.0, 0.0],
                    [axis_length, 0.0, 0.0],  # +X
                    [0.0, axis_length, 0.0],  # +Y
                    [0.0, 0.0, axis_length],  # +Z
                ]
            ).reshape(-1, 3)
            try:
                axis_points_c, _ = cv2.projectPoints(
                    axis_points_w,
                    self.rvec,
                    self.tvec,
                    self.mtx,
                    self.dist,
                )
            except Exception:
                return frame

            self.world_axis_c = []
            for i in range(4):
                self.world_axis_c.append(tuple(int(x) for x in axis_points_c[i].ravel()))

        origin = self.world_axis_c[0]
        x_tip = self.world_axis_c[1]
        y_tip = self.world_axis_c[2]
        z_tip = self.world_axis_c[3]

        origin_pt = self._to_cv_point(origin)
        x_tip_pt = self._to_cv_point(x_tip)
        y_tip_pt = self._to_cv_point(y_tip)
        z_tip_pt = self._to_cv_point(z_tip)
        if None in (origin_pt, x_tip_pt, y_tip_pt, z_tip_pt):
            # Recompute next frame if cached points become invalid.
            self.world_axis_c = None
            return frame

        # BGR colors: X=red, Y=green, Z=blue
        self._draw_arrow(frame, origin_pt, x_tip_pt, (0, 0, 255), thickness=2, tip_length=0.2)
        self._draw_arrow(frame, origin_pt, y_tip_pt, (0, 255, 0), thickness=2, tip_length=0.2)
        self._draw_arrow(frame, origin_pt, z_tip_pt, (255, 0, 0), thickness=2, tip_length=0.2)
        cv2.putText(
            frame,
            "X",
            (x_tip_pt[0] + 4, x_tip_pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            "Y",
            (y_tip_pt[0] + 4, y_tip_pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            "Z",
            (z_tip_pt[0] + 4, z_tip_pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        return frame

    def transform_w2c(self,points3d_world):
        return cv2.projectPoints(points3d_world, self.r, self.t, self.mtx, self.dist)[0].flatten()
    
    def compute_fov_on_ground(self, ground_z=0.0):
        """Compute and cache the field of view intersection with ground plane."""
        if self.fov_ground_pts is not None:
            return self.fov_ground_pts  # already computed
        if getattr(self, "r", None) is None or getattr(self, "t", None) is None:
            return None

        if self.frame_dims is None and self.frame is not None:
            self.frame_dims = self.frame.shape[:2]
        if self.frame_dims is None:
            return None

        fx, fy = self.mtx[0, 0], self.mtx[1, 1]
        height, width = self.frame_dims

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

        def _fallback_wedge_xy(radius):
            # Fallback for non-ground-referenced poses (e.g., auto calibration).
            # Uses camera forward direction projected into XY to build a visible FOV wedge.
            forward = self.r.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            heading_xy = np.array([forward[0], forward[1]], dtype=np.float64)
            norm_h = float(np.linalg.norm(heading_xy))
            if norm_h < 1e-9:
                heading_xy = np.array([1.0, 0.0], dtype=np.float64)
                norm_h = 1.0
            heading_xy /= norm_h

            half = float(fov_x) * 0.5
            c = float(np.cos(half))
            s = float(np.sin(half))
            rot_left = np.array([[c, -s], [s, c]], dtype=np.float64)
            rot_right = np.array([[c, s], [-s, c]], dtype=np.float64)
            left_xy = heading_xy @ rot_left.T
            right_xy = heading_xy @ rot_right.T

            left = np.array(
                [C[0] + radius * left_xy[0], C[1] + radius * left_xy[1], float(ground_z)],
                dtype=np.float64,
            )
            right = np.array(
                [C[0] + radius * right_xy[0], C[1] + radius * right_xy[1], float(ground_z)],
                dtype=np.float64,
            )
            return left, right

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

        dist_to_origin = float(np.linalg.norm(C))
        fallback_radius = max(1.0, dist_to_origin, float(self.world_axis_length) * 25.0)

        # choose the two ground points with largest lateral separation
        if len(pts) >= 2:
            dists = np.linalg.norm(pts[:, None, :2] - pts[None, :, :2], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            left, right = pts[i], pts[j]
        else:
            left, right = _fallback_wedge_xy(fallback_radius)
            self.fov_ground_pts = {"C": C, "left": left, "right": right}
            return self.fov_ground_pts

        left_vec = left - C
        right_vec = right - C
        if np.linalg.norm(left_vec) < 1e-9 or np.linalg.norm(right_vec) < 1e-9:
            left, right = _fallback_wedge_xy(fallback_radius)
            self.fov_ground_pts = {"C": C, "left": left, "right": right}
            return self.fov_ground_pts

        radius = max(1.0, dist_to_origin)
        left = C + left_vec / np.linalg.norm(left_vec) * radius
        right = C + right_vec / np.linalg.norm(right_vec) * radius

        self.fov_ground_pts = {"C": C, "left": left, "right": right} # Each is a 3D Point in World Coordinates
        return self.fov_ground_pts
