from .Camera import Camera
from .ModelPlus import ModelPlus
from .MCMOTUtils import MCMOTUtils
from .Ned2ArmController import Ned2ArmController
import time
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

class MCMOTracker:
    def __init__(
        self,
        model_path=None,
        camera_device_ids=None,
        camera_names=None,
        confidence_threshold=.25,
        tracker_yaml_path="./config/bytetrack.yaml",
        aruco_positions=None,
        display=False,
        calibration_method="aruco",
        calibration=None,
        mesh_display=True,
        mesh_update_interval_sec=3.0,
        mesh_export_path="./artifacts/scene_mesh.glb",
        overhead_update_interval_sec=2.0,
        arm_model_path=None,
        arm_inference_interval_sec=0.5,
        show_world_axis=True,
        world_axis_length=3.0,
    ):
        if calibration is not None:
            calibration_method = calibration

        self.parts_model_path = model_path
        self.parts_model = None
        self.parts_model_loaded = False
        self.parts_confidence_threshold = float(confidence_threshold)
        self.parts_nms_enabled = True
        self.parts_nms_iou_threshold = 0.7
        self.arm_model_path = arm_model_path
        self.arm_model = None
        self.arm_model_loaded = False
        self.arm_detections_by_camera = {}
        self.arm_confidence_threshold = float(confidence_threshold)
        self.arm_nms_enabled = True
        self.arm_nms_iou_threshold = 0.4
        self.arm_keypoint_names = ("base", "shoulder", "elbow", "wrist", "gripper")
        self.arm_keypoint_min_confidence = 0.10
        self.arm_keypoint_matches = []
        self.arm_detection_links_by_camera = {}
        self.arm_objects = {}
        self.arm_controller_links = {}
        self.next_arm_object_id = 0
        self.arm_base_link_distance_threshold = 25.0
        self.arm_link_max_missed_frames = 60
        self._arm_match_frame_index = 0
        self._last_arm_inference_time = 0.0
        self.ned2_arm_controller = None
        self._sync_legacy_parts_aliases()
        self.tracker_yaml_path = tracker_yaml_path
        self.camera_device_ids = list(camera_device_ids or [])
        self.camera_names = list(camera_names or [])
        self.aruco_positions = aruco_positions
        self.cameras = {}  # {camera_id: Camera}
        self.global_tracks = {}
        self.camera_tracks = {}  # {ctid: {track_id: CameraTrack}}
        self.next_ctid = 0
        self.next_gtid = 0
        self.next_track_id = 0
        self.overhead_plot = None
        self.display = display
        self.gui_available = MCMOTUtils.has_highgui()
        self.calibration_method = calibration_method.lower() if calibration_method else None
        if self.calibration_method not in {"aruco", "landmarks", "auto", None}:
            raise ValueError(
                f"Unknown calibration_method '{calibration_method}'. "
                "Expected 'aruco', 'landmarks', or 'auto'."
            )
        self.mesh_display = mesh_display
        self.mesh_update_interval_sec = float(mesh_update_interval_sec)
        self.mesh_export_path = mesh_export_path
        self.mesh_last_update_time = 0.0
        self.mesh_status = ""
        self.mesh_window_name = "Scene Mesh"
        self.scene_mesh = None
        self.mesh_visualizer = None
        self._open3d_module = None
        self._open3d_available = None
        self._warned_open3d_missing = False
        self.mesh_stereo_ready = False
        self.mesh_rect_size = None
        self.mesh_map0x = None
        self.mesh_map0y = None
        self.mesh_map1x = None
        self.mesh_map1y = None
        self.mesh_Q = None
        self.mesh_R_rel = None
        self.mesh_T_rel = None
        self.mesh_depth_min_disparity = 1.0
        self.mesh_depth_max = 5000.0
        self.mesh_max_points = 50000
        self.mesh_poisson_depth = 8
        self.mesh_triangle_budget = 120000
        self.show_world_axis = bool(show_world_axis)
        self.world_axis_length = float(world_axis_length)
        self.arm_overlay_color_bgr = (0, 165, 255)  # orange in BGR
        self.arm_axis_length = max(0.25, float(self.world_axis_length) * 0.75)
        self.set_overhead_update_interval(float(overhead_update_interval_sec))
        self.set_arm_inference_interval(float(arm_inference_interval_sec))
        self.mesh_stereo_matcher = None
        if self.mesh_display:
            block_size = 5
            num_disparities = 16 * 10
            self.mesh_stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size**2,
                P2=32 * 3 * block_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        if display is True and self.gui_available:
            cv2.namedWindow("Overhead View", cv2.WINDOW_NORMAL)
        elif display is True and not self.gui_available:
            print("OpenCV HighGUI unavailable; overhead OpenCV window disabled.")

    def _sync_legacy_parts_aliases(self):
        # Backward-compatible aliases used by older scripts/services.
        self.model_path = self.parts_model_path
        self.model = self.parts_model
        self.model_loaded = self.parts_model_loaded
        self.confidence_threshold = self.parts_confidence_threshold

    def select_cameras(self):
        # Show image from all available cameras
        # todo
        # selected_cameras = util.select_cameras(2)
        failures = []

        for camera_number, camera_device_id in enumerate(self.camera_device_ids):
            print(f"Initializing Camera {camera_number} - Port {camera_device_id} - {self.camera_names[camera_number]}")
            camera_name = self.camera_names[camera_number]
            try:
                self.cameras[camera_number] = Camera(
                    camera_number,
                    camera_device_id,
                    camera_name,
                    None,
                    self.parts_confidence_threshold,
                    self.tracker_yaml_path,
                    aruco_positions=self.aruco_positions,
                    display=self.display,
                    calibration_method=self.calibration_method,
                    show_world_axis=self.show_world_axis,
                    world_axis_length=self.world_axis_length,
                )
                print("    Completed Initialization")
            except Exception as exc:
                failures.append((camera_number, camera_device_id, camera_name, str(exc)))
                print(f"    Initialization failed: {exc}")

        if failures:
            details = "; ".join(
                [
                    f"cam#{num} id={cid} name={name}: {msg}"
                    for num, cid, name, msg in failures
                ]
            )
            raise RuntimeError(f"Camera initialization failed: {details}")

        if self.calibration_method == "auto":
            self.calibrate_auto()

        if self.parts_model_path is not None:
            self.set_parts_model(
                model_path=self.parts_model_path,
                confidence_threshold=self.parts_confidence_threshold,
                tracker_yaml_path=self.tracker_yaml_path,
            )
        if self.arm_model_path is not None:
            self.set_arm_model(self.arm_model_path)

    def _reset_camera_trackers(self):
        for camera in self.cameras.values():
            camera.model_plus = ModelPlus(
                model_path=None,
                confidence_threshold=self.parts_confidence_threshold,
                tracking_yaml_path=self.tracker_yaml_path,
            )

    def set_parts_model(
        self,
        model_path,
        confidence_threshold=None,
        tracker_yaml_path=None,
    ):
        if model_path == "LATEST":
            model_path = MCMOTUtils.find_latest_model()[0]

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if confidence_threshold is not None:
            self.parts_confidence_threshold = float(confidence_threshold)
            self.confidence_threshold = self.parts_confidence_threshold
        if tracker_yaml_path is not None:
            self.tracker_yaml_path = str(tracker_yaml_path)

        self.parts_model = YOLO(model_path, verbose=False)
        self.parts_model_path = model_path
        self.parts_model_loaded = True
        self._sync_legacy_parts_aliases()
        self.global_tracks = {}
        self._reset_camera_trackers()
        return {
            "parts_model_path": self.parts_model_path,
            "parts_model_loaded": self.parts_model_loaded,
            "model_path": self.parts_model_path,
            "model_loaded": self.parts_model_loaded,
            "parts_nms_enabled": bool(self.parts_nms_enabled),
            "parts_nms_iou_threshold": float(self.parts_nms_iou_threshold),
            "num_trackers": len(self.cameras),
        }

    def clear_parts_model(self):
        self.parts_model = None
        self.parts_model_loaded = False
        self.parts_model_path = None
        self._sync_legacy_parts_aliases()
        self.global_tracks = {}
        for camera in self.cameras.values():
            camera.detach_model()
        return {
            "parts_model_loaded": self.parts_model_loaded,
            "model_loaded": self.parts_model_loaded,
        }

    def set_parts_confidence_threshold(self, confidence_threshold):
        value = float(confidence_threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        self.parts_confidence_threshold = value
        self.confidence_threshold = self.parts_confidence_threshold
        applied_to = 0
        for camera in self.cameras.values():
            if camera.model_plus is not None:
                camera.model_plus.confidence_threshold = value
                applied_to += 1
        return {
            "parts_confidence_threshold": self.parts_confidence_threshold,
            "confidence_threshold": self.parts_confidence_threshold,
            "applied_model_count": applied_to,
            "parts_model_loaded": self.parts_model_loaded,
            "model_loaded": self.parts_model_loaded,
        }

    def set_parts_nms_options(self, enabled=None, iou_threshold=None):
        if enabled is not None:
            self.parts_nms_enabled = bool(enabled)
        if iou_threshold is not None:
            iou_value = float(iou_threshold)
            if iou_value < 0.0 or iou_value > 1.0:
                raise ValueError("iou_threshold must be between 0.0 and 1.0")
            self.parts_nms_iou_threshold = iou_value
        return {
            "parts_nms_enabled": bool(self.parts_nms_enabled),
            "parts_nms_iou_threshold": float(self.parts_nms_iou_threshold),
        }

    def set_arm_model(self, model_path, confidence_threshold=None):
        if model_path == "LATEST":
            model_path = MCMOTUtils.find_latest_model()[0]

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Arm model file not found: {model_path}")
        if confidence_threshold is not None:
            value = float(confidence_threshold)
            if value < 0.0 or value > 1.0:
                raise ValueError("arm confidence_threshold must be between 0.0 and 1.0")
            self.arm_confidence_threshold = value

        self.arm_model = YOLO(model_path, verbose=False)
        self.arm_model_path = model_path
        self.arm_model_loaded = True
        self.arm_detections_by_camera = {}
        self.arm_keypoint_matches = []
        self.arm_detection_links_by_camera = {}
        self.arm_objects = {}
        self.arm_controller_links = {}
        self.next_arm_object_id = 0
        self._arm_match_frame_index = 0
        self._last_arm_inference_time = 0.0
        return {
            "arm_model_path": self.arm_model_path,
            "arm_model_loaded": self.arm_model_loaded,
            "arm_confidence_threshold": float(self.arm_confidence_threshold),
            "arm_nms_enabled": bool(self.arm_nms_enabled),
            "arm_nms_iou_threshold": float(self.arm_nms_iou_threshold),
        }

    def clear_arm_model(self):
        self.arm_model = None
        self.arm_model_path = None
        self.arm_model_loaded = False
        self.arm_detections_by_camera = {}
        self.arm_keypoint_matches = []
        self.arm_detection_links_by_camera = {}
        self.arm_objects = {}
        self.arm_controller_links = {}
        self.next_arm_object_id = 0
        self._arm_match_frame_index = 0
        self._last_arm_inference_time = 0.0
        return {"arm_model_loaded": self.arm_model_loaded}

    def set_arm_inference_interval(self, interval_sec):
        value = float(interval_sec)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("arm_inference_interval_sec must be >= 0.0")
        self.arm_inference_interval_sec = value
        return {"arm_inference_interval_sec": self.arm_inference_interval_sec}

    def set_arm_confidence_threshold(self, confidence_threshold):
        value = float(confidence_threshold)
        if value < 0.0 or value > 1.0:
            raise ValueError("arm confidence_threshold must be between 0.0 and 1.0")
        self.arm_confidence_threshold = value
        # Force immediate re-inference with the new threshold.
        self._last_arm_inference_time = 0.0
        # Drop currently cached low-confidence detections so UI updates immediately.
        for camera_id, detections in list(self.arm_detections_by_camera.items()):
            filtered = []
            for detection in detections:
                conf = detection.get("confidence")
                if conf is None:
                    filtered.append(detection)
                    continue
                if float(conf) >= self.arm_confidence_threshold:
                    filtered.append(detection)
            self.arm_detections_by_camera[camera_id] = filtered
        self.arm_keypoint_matches = []
        self.arm_detection_links_by_camera = {}
        return {
            "arm_confidence_threshold": float(self.arm_confidence_threshold),
            "arm_model_loaded": bool(self.arm_model_loaded),
        }

    def set_arm_nms_options(self, enabled=None, iou_threshold=None):
        if enabled is not None:
            self.arm_nms_enabled = bool(enabled)
        if iou_threshold is not None:
            iou_value = float(iou_threshold)
            if iou_value < 0.0 or iou_value > 1.0:
                raise ValueError("arm iou_threshold must be between 0.0 and 1.0")
            self.arm_nms_iou_threshold = iou_value
        return {
            "arm_nms_enabled": bool(self.arm_nms_enabled),
            "arm_nms_iou_threshold": float(self.arm_nms_iou_threshold),
        }

    def connect_ned2_arm(
        self,
        ip_address,
        port=9090,
        auto_calibrate=True,
        disable_learning_mode=True,
        update_tool=True,
    ):
        if self.ned2_arm_controller is None:
            self.ned2_arm_controller = Ned2ArmController()
        data = self.ned2_arm_controller.connect(
            ip_address=ip_address,
            port=port,
            auto_calibrate=auto_calibrate,
            disable_learning_mode=disable_learning_mode,
            update_tool=update_tool,
        )
        return {"ned2_connected": True, **data}

    def disconnect_ned2_arm(self):
        if self.ned2_arm_controller is None:
            return {"ned2_connected": False}
        controller = self.ned2_arm_controller
        stale_links = [arm_id for arm_id, linked in self.arm_controller_links.items() if linked is controller]
        for arm_id in stale_links:
            self.arm_controller_links.pop(arm_id, None)
        data = self.ned2_arm_controller.disconnect()
        return {"ned2_connected": False, **data}

    def move_ned2_arm_robot_xyz(
        self,
        x,
        y,
        z,
        roll=None,
        pitch=None,
        yaw=None,
        frame="",
    ):
        if self.ned2_arm_controller is None:
            raise RuntimeError("Ned2 arm controller is not initialized.")
        pose = self.ned2_arm_controller.move_robot_xyz(
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            frame=frame,
        )
        return {"robot_pose": pose}

    def move_ned2_arm_global_xyz(
        self,
        x,
        y,
        z,
        roll=None,
        pitch=None,
        yaw=None,
        frame="",
    ):
        if self.ned2_arm_controller is None:
            raise RuntimeError("Ned2 arm controller is not initialized.")
        return self.ned2_arm_controller.move_global_xyz(
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            frame=frame,
        )

    def calibrate_ned2_arm_grid(
        self,
        observe_global_xyz_fn,
        x_values,
        y_values,
        z_values,
        orientation_rpy=None,
        settle_time_sec=0.35,
        allow_scale=True,
        require_all_points=False,
    ):
        if self.ned2_arm_controller is None:
            raise RuntimeError("Ned2 arm controller is not initialized.")
        return self.ned2_arm_controller.run_grid_calibration(
            observe_global_xyz_fn=observe_global_xyz_fn,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
            orientation_rpy=orientation_rpy,
            settle_time_sec=settle_time_sec,
            allow_scale=allow_scale,
            require_all_points=require_all_points,
        )

    def set_ned2_transform(
        self,
        rotation_robot_to_global,
        translation_robot_to_global,
        scale_robot_to_global=1.0,
        source="manual",
        rmse=None,
    ):
        if self.ned2_arm_controller is None:
            self.ned2_arm_controller = Ned2ArmController()
        return self.ned2_arm_controller.set_transform(
            rotation_robot_to_global=rotation_robot_to_global,
            translation_robot_to_global=translation_robot_to_global,
            scale_robot_to_global=scale_robot_to_global,
            source=source,
            rmse=rmse,
        )

    def get_ned2_transform(self):
        if self.ned2_arm_controller is None:
            return {"transform_ready": False}
        return self.ned2_arm_controller.get_transform_dict()

    def get_active_arm_objects(self, include_missed=False):
        arm_objects = {}
        for arm_id, arm_obj in self.arm_objects.items():
            missed_frames = int(arm_obj.get("missed_frames", 0))
            if not include_missed and missed_frames > 0:
                continue
            payload = dict(arm_obj)
            payload["controller_linked"] = bool(int(arm_id) in self.arm_controller_links)
            arm_objects[int(arm_id)] = payload
        return arm_objects

    def set_arm_object_no_control(self, arm_id, no_control=True):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            raise ValueError(f"Unknown arm_id {arm_id}.")
        arm_obj["no_control"] = bool(no_control)
        if arm_obj["no_control"]:
            self.arm_controller_links.pop(arm_id, None)
        return {"arm_id": arm_id, "no_control": bool(arm_obj["no_control"])}

    def set_arm_object_calibrated(self, arm_id, calibrated=True):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            raise ValueError(f"Unknown arm_id {arm_id}.")
        arm_obj["calibrated"] = bool(calibrated)
        return {"arm_id": arm_id, "calibrated": bool(arm_obj["calibrated"])}

    def set_arm_object_in_scene(self, arm_id, in_scene=True):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            raise ValueError(f"Unknown arm_id {arm_id}.")
        arm_obj["added_to_scene"] = bool(in_scene)
        return {"arm_id": arm_id, "added_to_scene": bool(arm_obj["added_to_scene"])}

    @staticmethod
    def _normalize_xyz(point_xyz):
        try:
            point = np.asarray(point_xyz, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if point.size != 3 or not np.isfinite(point).all():
            return None
        return point

    def _get_arm_base_marker_xyz(self, arm_obj):
        for key in ("user_base_xyz", "base_xyz"):
            point = self._normalize_xyz(arm_obj.get(key))
            if point is not None:
                return point
        return None

    def _get_arm_link_reference_xyz(self, arm_obj):
        if arm_obj is None:
            return None
        if bool(arm_obj.get("added_to_scene", False)):
            user_base = self._normalize_xyz(arm_obj.get("user_base_xyz"))
            if user_base is not None:
                return user_base
        return self._normalize_xyz(arm_obj.get("base_xyz"))

    def _get_arm_rotation_robot_to_global(self, arm_id):
        controller = self.arm_controller_links.get(int(arm_id))
        if controller is None or not bool(getattr(controller, "transform_ready", False)):
            return None
        try:
            transform = controller.get_transform_dict()
            rotation = np.asarray(
                transform.get("rotation_robot_to_global"),
                dtype=np.float64,
            ).reshape(3, 3)
        except Exception:
            return None
        if not np.isfinite(rotation).all():
            return None
        return rotation

    def set_arm_user_base_xyz(self, arm_id, base_xyz=None):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            raise ValueError(f"Unknown arm_id {arm_id}.")

        if base_xyz is None:
            # Prefer the latest tracked base when no explicit point is provided.
            base_xyz = arm_obj.get("base_xyz", arm_obj.get("user_base_xyz"))
        base_point = self._normalize_xyz(base_xyz)
        if base_point is None:
            raise ValueError("base_xyz must be a finite [x, y, z] point.")

        arm_obj["user_base_xyz"] = base_point.astype(float).tolist()
        arm_obj.setdefault("arm_origin_xyz", arm_obj["user_base_xyz"])
        return {
            "arm_id": arm_id,
            "user_base_xyz": arm_obj["user_base_xyz"],
        }

    def sync_arm_origin_from_controller(self, arm_id):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            raise ValueError(f"Unknown arm_id {arm_id}.")

        controller = self.arm_controller_links.get(arm_id)
        if controller is None or not bool(getattr(controller, "transform_ready", False)):
            return {"arm_id": arm_id, "arm_origin_xyz": arm_obj.get("arm_origin_xyz")}

        try:
            origin_xyz = controller.robot_to_global_xyz([0.0, 0.0, 0.0])
        except Exception:
            return {"arm_id": arm_id, "arm_origin_xyz": arm_obj.get("arm_origin_xyz")}

        origin_point = self._normalize_xyz(origin_xyz)
        if origin_point is None:
            return {"arm_id": arm_id, "arm_origin_xyz": arm_obj.get("arm_origin_xyz")}

        arm_obj["arm_origin_xyz"] = origin_point.astype(float).tolist()
        return {"arm_id": arm_id, "arm_origin_xyz": arm_obj["arm_origin_xyz"]}

    def _project_global_xyz_to_camera(self, camera_number, point_xyz):
        camera = self.cameras.get(int(camera_number))
        if camera is None:
            return None

        point = self._normalize_xyz(point_xyz)
        if point is None:
            return None

        r_input = getattr(camera, "rvec", None)
        t_input = getattr(camera, "tvec", None)
        if r_input is None:
            r_input = getattr(camera, "r", None)
        if t_input is None:
            t_input = getattr(camera, "t", None)
        required = (r_input, t_input, getattr(camera, "mtx", None), getattr(camera, "dist", None))
        if any(v is None for v in required):
            return None

        try:
            projected, _ = cv2.projectPoints(
                point.reshape(1, 1, 3),
                r_input,
                t_input,
                camera.mtx,
                camera.dist,
            )
        except Exception:
            return None
        return Camera._to_cv_point(projected.reshape(-1)[:2])

    @staticmethod
    def _draw_cross_marker(frame, center_px, color, size=8, thickness=2):
        if center_px is None:
            return
        cx, cy = int(center_px[0]), int(center_px[1])
        s = int(max(2, size))
        cv2.line(frame, (cx - s, cy - s), (cx + s, cy + s), color, int(max(1, thickness)))
        cv2.line(frame, (cx - s, cy + s), (cx + s, cy - s), color, int(max(1, thickness)))

    def _draw_arm_axes_on_frame(self, frame, camera_number, origin_xyz, rotation_robot_to_global):
        camera = self.cameras.get(int(camera_number))
        if camera is None:
            return

        origin_px = self._project_global_xyz_to_camera(camera_number, origin_xyz)
        if origin_px is None:
            return

        axis_length = float(self.arm_axis_length)
        axis_names = ("x", "y", "z")
        for axis_idx, axis_name in enumerate(axis_names):
            axis_tip_xyz = np.asarray(origin_xyz, dtype=np.float64) + axis_length * np.asarray(
                rotation_robot_to_global[:, axis_idx],
                dtype=np.float64,
            )
            tip_px = self._project_global_xyz_to_camera(camera_number, axis_tip_xyz)
            if tip_px is None:
                continue
            camera._draw_arrow(
                frame,
                origin_px,
                tip_px,
                self.arm_overlay_color_bgr,
                thickness=2,
                tip_length=0.2,
            )
            cv2.putText(
                frame,
                axis_name,
                (tip_px[0] + 4, tip_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.arm_overlay_color_bgr,
                1,
            )

    def _annotate_arm_origin_overlays(self, frame, camera_number):
        for arm_id in sorted(self.arm_objects.keys()):
            arm_obj = self.arm_objects.get(arm_id, {})
            if not bool(arm_obj.get("added_to_scene", False)):
                continue
            base_xyz = self._get_arm_base_marker_xyz(arm_obj)
            if base_xyz is not None:
                base_px = self._project_global_xyz_to_camera(camera_number, base_xyz)
                if base_px is not None:
                    self._draw_cross_marker(
                        frame,
                        base_px,
                        self.arm_overlay_color_bgr,
                        size=8,
                        thickness=2,
                    )
                    cv2.putText(
                        frame,
                        f"{base_xyz[0]:.1f}, {base_xyz[1]:.1f}, {base_xyz[2]:.1f}",
                        (base_px[0] + 8, base_px[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        self.arm_overlay_color_bgr,
                        1,
                    )

            if not bool(arm_obj.get("calibrated", False)):
                continue

            axis_origin_xyz = self._normalize_xyz(arm_obj.get("arm_origin_xyz"))
            if axis_origin_xyz is None:
                axis_origin_xyz = base_xyz
            if axis_origin_xyz is None:
                continue

            rotation = self._get_arm_rotation_robot_to_global(arm_id)
            if rotation is None:
                rotation = np.eye(3, dtype=np.float64)

            self._draw_arm_axes_on_frame(
                frame=frame,
                camera_number=camera_number,
                origin_xyz=axis_origin_xyz,
                rotation_robot_to_global=rotation,
            )

    def get_arm_gripper_global_xyz(self, arm_id=None):
        if arm_id is None:
            active = self.get_active_arm_objects(include_missed=False)
            if not active:
                return None
            arm_id = sorted(active.keys())[0]
            arm_obj = active[arm_id]
        else:
            arm_id = int(arm_id)
            arm_obj = self.arm_objects.get(arm_id)
            if arm_obj is None:
                return None

        keypoints_3d = arm_obj.get("keypoints_3d", {})
        gripper_xyz = keypoints_3d.get("gripper")
        if gripper_xyz is None:
            return None
        point = np.asarray(gripper_xyz, dtype=np.float64).reshape(-1)
        if point.size != 3 or not np.isfinite(point).all():
            return None
        return point.astype(float).tolist()

    def link_ned2_controller_to_arm(self, arm_id, controller=None):
        arm_id = int(arm_id)
        if arm_id not in self.arm_objects:
            raise ValueError(f"Unknown arm_id {arm_id}.")
        arm_obj = self.arm_objects[arm_id]
        if bool(arm_obj.get("no_control", False)):
            raise RuntimeError(f"arm_id {arm_id} is marked as no_control and cannot be linked.")
        if controller is None:
            controller = self.ned2_arm_controller
        if controller is None:
            raise RuntimeError("No Ned2 controller available to link.")
        self.arm_controller_links[arm_id] = controller
        return {"arm_id": arm_id, "controller_linked": True}

    def unlink_ned2_controller_from_arm(self, arm_id):
        arm_id = int(arm_id)
        self.arm_controller_links.pop(arm_id, None)
        return {"arm_id": arm_id, "controller_linked": False}

    def move_linked_arm_global_xyz(
        self,
        arm_id,
        x,
        y,
        z,
        roll=None,
        pitch=None,
        yaw=None,
        frame="",
    ):
        arm_id = int(arm_id)
        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is not None and bool(arm_obj.get("no_control", False)):
            raise RuntimeError(f"arm_id {arm_id} is marked as no_control and cannot be moved.")
        controller = self.arm_controller_links.get(arm_id)
        if controller is None:
            raise RuntimeError(f"arm_id {arm_id} is not linked to a Ned2 controller.")
        result = controller.move_global_xyz(
            x,
            y,
            z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            frame=frame,
        )
        return {"arm_id": arm_id, **result}

    def force_arm_detection_update(self, rematch=True):
        """
        Force a fresh arm detection pass immediately, bypassing interval throttling.
        """
        self.capture_cameras_frames()
        self._last_arm_inference_time = 0.0
        self.update_cameras_tracks()
        if rematch:
            self.match_arm_keypoints_between_cameras()
        return {
            "arm_detections_by_camera": {
                int(cid): int(len(dets))
                for cid, dets in self.arm_detections_by_camera.items()
            },
            "arm_objects": self.get_active_arm_objects(include_missed=True),
        }

    def object_pick_linked_arm(
        self,
        arm_id,
        target_global_xyz,
        orientation_rpy=None,
        approach_height_in=2.0,
        descent_checkpoints_in=(1.0, 0.5, 0.2, 0.0),
        verify_tolerance_in=1.0,
        verify_retries=4,
        verify_pause_sec=0.12,
        move_settle_sec=0.2,
        lift_height_in=2.0,
        lift_steps=4,
        require_vision=True,
        frame="",
    ):
        arm_id = int(arm_id)
        target_global = np.asarray(target_global_xyz, dtype=np.float64).reshape(-1)
        if target_global.size != 3 or not np.isfinite(target_global).all():
            raise ValueError("target_global_xyz must be finite [x, y, z].")

        controller = self.arm_controller_links.get(arm_id)
        if controller is None:
            raise RuntimeError(f"arm_id {arm_id} is not linked to a Ned2 controller.")

        def _force_vision_update():
            self.force_arm_detection_update(rematch=True)

        def _observe_gripper_global():
            self.force_arm_detection_update(rematch=True)
            return self.get_arm_gripper_global_xyz(arm_id)

        result = controller.object_pick_global(
            x=float(target_global[0]),
            y=float(target_global[1]),
            z=float(target_global[2]),
            observe_gripper_global_xyz_fn=_observe_gripper_global,
            force_vision_update_fn=_force_vision_update,
            orientation_rpy=orientation_rpy,
            frame=frame,
            approach_height_in=approach_height_in,
            descent_checkpoints_in=descent_checkpoints_in,
            verify_tolerance_in=verify_tolerance_in,
            verify_retries=verify_retries,
            verify_pause_sec=verify_pause_sec,
            move_settle_sec=move_settle_sec,
            lift_height_in=lift_height_in,
            lift_steps=lift_steps,
            require_vision=require_vision,
        )
        return {"arm_id": arm_id, **result}

    # Backward-compatible APIs for existing callers.
    def set_detection_model(
        self,
        model_path,
        confidence_threshold=None,
        tracker_yaml_path=None,
    ):
        return self.set_parts_model(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            tracker_yaml_path=tracker_yaml_path,
        )

    def clear_detection_model(self):
        return self.clear_parts_model()

    def set_confidence_threshold(self, confidence_threshold):
        return self.set_parts_confidence_threshold(confidence_threshold)

    def set_overhead_update_interval(self, interval_sec):
        value = float(interval_sec)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("overhead_update_interval_sec must be >= 0.0")
        self.overhead_update_interval_sec = value
        return {"overhead_update_interval_sec": self.overhead_update_interval_sec}

    def calibrate_auto(self, baseline=1.0):
        if len(self.cameras) < 2:
            raise RuntimeError("Auto calibration requires at least two cameras.")

        ref_id = sorted(self.cameras.keys())[0]
        ref_camera = self.cameras[ref_id]
        ref_camera.set_extrinsics_from_rt(np.eye(3), np.zeros((3, 1)), world_calibrated=False)

        print(f"Auto calibration reference camera: {ref_camera.camera_name} (id={ref_id})")
        failures = []
        for camera_id in sorted(self.cameras.keys()):
            if camera_id == ref_id:
                continue
            camera = self.cameras[camera_id]
            ok = camera.cal_auto(ref_camera, baseline=baseline)
            if not ok:
                failures.append((camera_id, camera.camera_name))

        if failures:
            fail_text = ", ".join([f"id={cid} ({name})" for cid, name in failures])
            print(
                "WARNING: Auto calibration failed for "
                f"{fail_text}. Global triangulation/mesh may be unavailable for those cameras."
            )
            if self.mesh_display:
                self.mesh_display = False
                print("Mesh display disabled because auto calibration is incomplete.")
    
    def capture_cameras_frames(self):
        for camera in self.cameras.values():
            camera.capture_frame()

    def update_cameras_tracks(self):
        for camera in self.cameras.values():
            camera.fps_counter.tick()

        batch_camera_ids = []
        batch_frames = []
        for camera_id in sorted(self.cameras.keys()):
            camera = self.cameras[camera_id]
            if camera.frame is None:
                if camera.model_plus is not None:
                    camera.model_plus.frame = None
                    camera.model_plus.detections = None
                    camera.model_plus.detections_tracked = None
                self.arm_detections_by_camera.pop(camera_id, None)
                continue
            if self.parts_model_loaded and camera.model_plus is None:
                camera.model_plus = ModelPlus(
                    model_path=None,
                    confidence_threshold=self.parts_confidence_threshold,
                    tracking_yaml_path=self.tracker_yaml_path,
                )
            batch_camera_ids.append(camera_id)
            batch_frames.append(camera.frame)

        if not batch_frames:
            return

        if self.parts_model_loaded and self.parts_model is not None:
            try:
                results = self.parts_model(
                    batch_frames,
                    conf=self.parts_confidence_threshold,
                    iou=self.parts_nms_iou_threshold,
                    agnostic_nms=bool(self.parts_nms_enabled),
                    nms=bool(self.parts_nms_enabled),
                    verbose=False,
                )
            except TypeError:
                results = self.parts_model(
                    batch_frames,
                    conf=self.parts_confidence_threshold,
                    iou=self.parts_nms_iou_threshold,
                    agnostic_nms=bool(self.parts_nms_enabled),
                )

            for camera_id, frame, result in zip(batch_camera_ids, batch_frames, results):
                camera = self.cameras[camera_id]
                camera.model_plus.update_from_result(result, frame)

        self._maybe_update_arm_detections(batch_camera_ids, batch_frames)

    def _maybe_update_arm_detections(self, batch_camera_ids, batch_frames):
        if not self.arm_model_loaded or self.arm_model is None:
            return

        now = time.time()
        if (
            self._last_arm_inference_time > 0.0
            and (now - self._last_arm_inference_time) < self.arm_inference_interval_sec
        ):
            return

        try:
            results = self.arm_model(
                batch_frames,
                conf=self.arm_confidence_threshold,
                iou=self.arm_nms_iou_threshold,
                agnostic_nms=bool(self.arm_nms_enabled),
                nms=bool(self.arm_nms_enabled),
                verbose=False,
            )
        except TypeError:
            results = self.arm_model(
                batch_frames,
                conf=self.arm_confidence_threshold,
                iou=self.arm_nms_iou_threshold,
                agnostic_nms=bool(self.arm_nms_enabled),
            )

        for camera_id, result in zip(batch_camera_ids, results):
            self.arm_detections_by_camera[camera_id] = self._extract_arm_detections(result)
        self._last_arm_inference_time = now

    def _extract_arm_detections(self, result):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = getattr(boxes, "xyxy", None)
        if xyxy is None:
            return []
        confs = getattr(boxes, "conf", None)
        class_ids = getattr(boxes, "cls", None)
        keypoints_obj = getattr(result, "keypoints", None)
        names = getattr(result, "names", None)
        if names is None and self.arm_model is not None:
            names = getattr(self.arm_model, "names", None)
        names = names or {}

        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        confs_np = (
            confs.detach().cpu().numpy()
            if (confs is not None and hasattr(confs, "detach"))
            else np.asarray(confs)
            if confs is not None
            else None
        )
        class_ids_np = (
            class_ids.detach().cpu().numpy()
            if (class_ids is not None and hasattr(class_ids, "detach"))
            else np.asarray(class_ids)
            if class_ids is not None
            else None
        )
        keypoints_xy_np = None
        keypoints_conf_np = None
        if keypoints_obj is not None:
            keypoints_xy = getattr(keypoints_obj, "xy", None)
            keypoints_conf = getattr(keypoints_obj, "conf", None)
            if keypoints_xy is not None:
                keypoints_xy_np = (
                    keypoints_xy.detach().cpu().numpy()
                    if hasattr(keypoints_xy, "detach")
                    else np.asarray(keypoints_xy)
                )
            if keypoints_conf is not None:
                keypoints_conf_np = (
                    keypoints_conf.detach().cpu().numpy()
                    if hasattr(keypoints_conf, "detach")
                    else np.asarray(keypoints_conf)
                )

        detections = []
        for idx, box in enumerate(xyxy_np):
            class_id = int(class_ids_np[idx]) if class_ids_np is not None else -1
            conf = float(confs_np[idx]) if confs_np is not None else None
            if conf is not None and conf < self.arm_confidence_threshold:
                continue
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            keypoints = {}
            if keypoints_xy_np is not None and idx < len(keypoints_xy_np):
                kp_xy = np.asarray(keypoints_xy_np[idx], dtype=np.float64)
                kp_conf = None
                if keypoints_conf_np is not None and idx < len(keypoints_conf_np):
                    kp_conf = np.asarray(keypoints_conf_np[idx], dtype=np.float64)
                max_kp = min(len(self.arm_keypoint_names), kp_xy.shape[0])
                for kp_idx in range(max_kp):
                    x = float(kp_xy[kp_idx, 0])
                    y = float(kp_xy[kp_idx, 1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    confidence = None
                    if kp_conf is not None and kp_idx < len(kp_conf):
                        confidence = float(kp_conf[kp_idx])
                        if (
                            np.isfinite(confidence)
                            and confidence < self.arm_keypoint_min_confidence
                        ):
                            continue
                    keypoints[self.arm_keypoint_names[kp_idx]] = {
                        "xy": [x, y],
                        "confidence": confidence,
                    }
            detections.append(
                {
                    "xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "class_id": class_id,
                    "confidence": conf,
                    "label": str(label),
                    "keypoints": keypoints,
                }
            )
        return detections

    def _get_arm_detection_keypoint_xy(self, detection, keypoint_name):
        keypoints = detection.get("keypoints", {})
        keypoint = keypoints.get(keypoint_name)
        if keypoint is None:
            return None
        xy = keypoint.get("xy")
        if xy is None or len(xy) != 2:
            return None
        x = float(xy[0])
        y = float(xy[1])
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return np.array([x, y], dtype=np.float64)

    def _triangulate_arm_keypoint(self, camera0, camera1, point0_xy, point1_xy):
        if camera0.projMatrix is None or camera1.projMatrix is None:
            return None, None

        p0 = np.asarray(point0_xy, dtype=np.float64).reshape(2, 1)
        p1 = np.asarray(point1_xy, dtype=np.float64).reshape(2, 1)
        try:
            point4d = cv2.triangulatePoints(
                camera0.projMatrix,
                camera1.projMatrix,
                p0,
                p1,
            )
        except Exception:
            return None, None

        if point4d is None or point4d.shape[0] < 4:
            return None, None
        w = float(point4d[3, 0])
        if abs(w) < 1e-9 or not np.isfinite(w):
            return None, None

        point3d = (point4d[:3, 0] / w).astype(np.float64)
        if not np.isfinite(point3d).all():
            return None, None

        homogeneous = np.append(point3d, 1.0)
        try:
            reproj0_h = camera0.projMatrix @ homogeneous
            reproj1_h = camera1.projMatrix @ homogeneous
        except Exception:
            return point3d, None

        z0 = float(reproj0_h[2])
        z1 = float(reproj1_h[2])
        if abs(z0) < 1e-9 or abs(z1) < 1e-9:
            return point3d, None

        reproj0 = np.array([reproj0_h[0] / z0, reproj0_h[1] / z0], dtype=np.float64)
        reproj1 = np.array([reproj1_h[0] / z1, reproj1_h[1] / z1], dtype=np.float64)
        err = float(np.linalg.norm(reproj0 - p0.ravel()) + np.linalg.norm(reproj1 - p1.ravel()))
        return point3d, err

    def image_xy_to_global_xyz(
        self,
        cam0_xy,
        cam1_xy,
        camera0_number=0,
        camera1_number=1,
        include_reprojection_error=False,
    ):
        """
        Triangulate a global XYZ point from one image XY point in each camera.

        Args:
            cam0_xy: (x, y) in camera0 image coordinates (pixels)
            cam1_xy: (x, y) in camera1 image coordinates (pixels)
            camera0_number: camera index for first view (default 0)
            camera1_number: camera index for second view (default 1)
            include_reprojection_error: include reprojection error in return payload

        Returns:
            [x, y, z] or {"global_xyz":[x,y,z], "reprojection_error":err}
        """
        camera0_number = int(camera0_number)
        camera1_number = int(camera1_number)
        if camera0_number not in self.cameras or camera1_number not in self.cameras:
            raise ValueError(
                f"Camera numbers {camera0_number} and/or {camera1_number} are not active."
            )

        point0 = np.asarray(cam0_xy, dtype=np.float64).reshape(-1)
        point1 = np.asarray(cam1_xy, dtype=np.float64).reshape(-1)
        if point0.size != 2 or point1.size != 2:
            raise ValueError("cam0_xy and cam1_xy must each contain exactly two values: [x, y].")
        if not np.isfinite(point0).all() or not np.isfinite(point1).all():
            raise ValueError("cam0_xy and cam1_xy must be finite numeric values.")

        camera0 = self.cameras[camera0_number]
        camera1 = self.cameras[camera1_number]
        point3d, reproj_error = self._triangulate_arm_keypoint(
            camera0,
            camera1,
            point0,
            point1,
        )
        if point3d is None:
            raise RuntimeError(
                "Triangulation failed. Ensure both cameras are calibrated with valid projection matrices."
            )

        global_xyz = point3d.astype(float).tolist()
        if include_reprojection_error:
            return {
                "global_xyz": global_xyz,
                "reprojection_error": None if reproj_error is None else float(reproj_error),
            }
        return global_xyz

    def resolve_or_create_arm_from_detection_pair(
        self,
        cam0_detection_index,
        cam1_detection_index,
        max_pair_cost=50.0,
        prefer_existing_within_threshold=True,
        create_new_when_linked_to_added=False,
    ):
        cam0_idx = int(cam0_detection_index)
        cam1_idx = int(cam1_detection_index)

        if 0 not in self.cameras or 1 not in self.cameras:
            raise RuntimeError("Arm pair resolution requires camera 0 and camera 1.")
        camera0 = self.cameras[0]
        camera1 = self.cameras[1]
        if camera0.projMatrix is None or camera1.projMatrix is None:
            raise RuntimeError("Cameras must be calibrated before resolving arm pair.")

        detections0 = self.arm_detections_by_camera.get(0, [])
        detections1 = self.arm_detections_by_camera.get(1, [])
        if cam0_idx < 0 or cam0_idx >= len(detections0):
            raise ValueError(f"Invalid cam0 detection index {cam0_idx}.")
        if cam1_idx < 0 or cam1_idx >= len(detections1):
            raise ValueError(f"Invalid cam1 detection index {cam1_idx}.")

        detection0 = detections0[cam0_idx]
        detection1 = detections1[cam1_idx]
        base0 = self._get_arm_detection_keypoint_xy(detection0, "base")
        base1 = self._get_arm_detection_keypoint_xy(detection1, "base")
        if base0 is None or base1 is None:
            raise RuntimeError("Selected detections must include base keypoints in both cameras.")

        base3d, base_err = self._triangulate_arm_keypoint(camera0, camera1, base0, base1)
        if base3d is None:
            raise RuntimeError("Unable to triangulate selected arm base keypoints.")
        if base_err is None:
            base_err = 0.0

        keypoints_3d = {}
        total_error = 0.0
        num_matched_keypoints = 0
        for keypoint_name in self.arm_keypoint_names:
            point0 = self._get_arm_detection_keypoint_xy(detection0, keypoint_name)
            point1 = self._get_arm_detection_keypoint_xy(detection1, keypoint_name)
            if point0 is None or point1 is None:
                continue
            point3d, reproj_error = self._triangulate_arm_keypoint(
                camera0,
                camera1,
                point0,
                point1,
            )
            if point3d is None:
                continue
            keypoints_3d[keypoint_name] = point3d.astype(float).tolist()
            if reproj_error is not None:
                total_error += float(reproj_error)
            num_matched_keypoints += 1

        if "base" not in keypoints_3d:
            keypoints_3d["base"] = base3d.astype(float).tolist()
            num_matched_keypoints += 1
        if num_matched_keypoints <= 0:
            raise RuntimeError("Unable to triangulate enough keypoints for selected arm pair.")

        avg_error = total_error / max(1, num_matched_keypoints)
        pair_cost = float(avg_error + 0.25 * float(base_err))
        if pair_cost > float(max_pair_cost):
            raise RuntimeError(
                f"Selected detections appear inconsistent between cameras (pair_cost={pair_cost:.2f})."
            )

        arm0 = self.arm_detection_links_by_camera.get(0, {}).get(cam0_idx)
        arm1 = self.arm_detection_links_by_camera.get(1, {}).get(cam1_idx)
        if arm0 is not None and arm1 is not None and int(arm0) != int(arm1):
            raise RuntimeError(
                f"Selected detections belong to different arm objects (cam0:{arm0}, cam1:{arm1})."
            )

        arm_id = int(arm0 if arm0 is not None else arm1) if (arm0 is not None or arm1 is not None) else None
        if arm_id is not None and bool(create_new_when_linked_to_added):
            linked_obj = self.arm_objects.get(int(arm_id), {})
            if bool(linked_obj.get("added_to_scene", False)):
                # In explicit "add arm" flows, allow a fresh arm object if the selected pair
                # is currently (mis)linked to an already added arm.
                arm_id = None
        base_point = np.asarray(keypoints_3d["base"], dtype=np.float64)

        if arm_id is None:
            if bool(prefer_existing_within_threshold):
                best_arm_id = None
                best_distance = None
                for existing_arm_id, arm_obj in self.arm_objects.items():
                    ref_point = self._get_arm_link_reference_xyz(arm_obj)
                    if ref_point is None:
                        continue
                    distance = float(np.linalg.norm(ref_point - base_point))
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_arm_id = existing_arm_id
                if (
                    best_arm_id is not None
                    and best_distance is not None
                    and best_distance <= self.arm_base_link_distance_threshold
                ):
                    arm_id = int(best_arm_id)

            if arm_id is None:
                arm_id = int(self.next_arm_object_id)
                self.next_arm_object_id += 1
                self.arm_objects[arm_id] = {
                    "arm_id": arm_id,
                    "no_control": False,
                    "calibrated": False,
                    "added_to_scene": False,
                }

        arm_obj = self.arm_objects.get(arm_id)
        if arm_obj is None:
            arm_obj = {
                "arm_id": int(arm_id),
                "no_control": False,
                "calibrated": False,
                "added_to_scene": False,
            }
            self.arm_objects[int(arm_id)] = arm_obj

        no_control = bool(arm_obj.get("no_control", False))
        calibrated = bool(arm_obj.get("calibrated", False))
        added_to_scene = bool(arm_obj.get("added_to_scene", False))
        user_base_xyz = self._normalize_xyz(arm_obj.get("user_base_xyz"))
        if user_base_xyz is None:
            user_base_xyz = np.asarray(base3d, dtype=np.float64).reshape(3)
        arm_origin_xyz = self._normalize_xyz(arm_obj.get("arm_origin_xyz"))
        if arm_origin_xyz is None:
            arm_origin_xyz = np.asarray(user_base_xyz, dtype=np.float64).reshape(3)
        base_prev = arm_obj.get("base_xyz")
        if base_prev is not None:
            base_prev_np = np.asarray(base_prev, dtype=np.float64).reshape(-1)
            if base_prev_np.size == 3 and np.isfinite(base_prev_np).all():
                base_point = 0.8 * base_prev_np + 0.2 * base_point

        arm_obj.update(
            {
                "arm_id": int(arm_id),
                "base_xyz": base_point.astype(float).tolist(),
                "keypoints_3d": keypoints_3d,
                "cam0_detection_index": int(cam0_idx),
                "cam1_detection_index": int(cam1_idx),
                "pair_cost": float(pair_cost),
                "last_seen_frame": int(self._arm_match_frame_index),
                "missed_frames": 0,
                "no_control": no_control,
                "calibrated": calibrated,
                "added_to_scene": added_to_scene,
                "user_base_xyz": user_base_xyz.astype(float).tolist(),
                "arm_origin_xyz": arm_origin_xyz.astype(float).tolist(),
            }
        )

        self.arm_detection_links_by_camera.setdefault(0, {})[int(cam0_idx)] = int(arm_id)
        self.arm_detection_links_by_camera.setdefault(1, {})[int(cam1_idx)] = int(arm_id)

        return {
            "arm_id": int(arm_id),
            "cam0_detection_index": int(cam0_idx),
            "cam1_detection_index": int(cam1_idx),
            "pair_cost": float(pair_cost),
            "num_keypoints": int(num_matched_keypoints),
        }

    def _link_matched_arms(self, matched_pairs):
        updated_ids = set()
        used_existing_ids = set()

        for pair in matched_pairs:
            base_point = np.asarray(pair["keypoints_3d"]["base"], dtype=np.float64)
            best_arm_id = None
            best_distance = None
            for arm_id, arm_obj in self.arm_objects.items():
                if arm_id in used_existing_ids:
                    continue
                ref_point = self._get_arm_link_reference_xyz(arm_obj)
                if ref_point is None:
                    continue
                distance = float(np.linalg.norm(ref_point - base_point))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_arm_id = arm_id

            if (
                best_arm_id is None
                or best_distance is None
                or best_distance > self.arm_base_link_distance_threshold
            ):
                arm_id = self.next_arm_object_id
                self.next_arm_object_id += 1
                self.arm_objects[arm_id] = {
                    "arm_id": arm_id,
                    "no_control": False,
                    "calibrated": False,
                    "added_to_scene": False,
                }
            else:
                arm_id = best_arm_id

            arm_obj = self.arm_objects[arm_id]
            no_control = bool(arm_obj.get("no_control", False))
            calibrated = bool(arm_obj.get("calibrated", False))
            added_to_scene = bool(arm_obj.get("added_to_scene", False))
            user_base_xyz = self._normalize_xyz(arm_obj.get("user_base_xyz"))
            if user_base_xyz is None:
                user_base_xyz = np.asarray(base_point, dtype=np.float64).reshape(3)
            arm_origin_xyz = self._normalize_xyz(arm_obj.get("arm_origin_xyz"))
            if arm_origin_xyz is None:
                arm_origin_xyz = np.asarray(user_base_xyz, dtype=np.float64).reshape(3)
            base_prev = arm_obj.get("base_xyz")
            if base_prev is not None:
                base_prev_np = np.asarray(base_prev, dtype=np.float64).reshape(-1)
                if base_prev_np.size == 3 and np.isfinite(base_prev_np).all():
                    base_point = 0.8 * base_prev_np + 0.2 * base_point

            arm_obj.update(
                {
                    "arm_id": int(arm_id),
                    "base_xyz": base_point.astype(float).tolist(),
                    "keypoints_3d": pair["keypoints_3d"],
                    "cam0_detection_index": int(pair["cam0_detection_index"]),
                    "cam1_detection_index": int(pair["cam1_detection_index"]),
                    "pair_cost": float(pair["cost"]),
                    "last_seen_frame": int(self._arm_match_frame_index),
                    "missed_frames": 0,
                    "no_control": no_control,
                    "calibrated": calibrated,
                    "added_to_scene": added_to_scene,
                    "user_base_xyz": user_base_xyz.astype(float).tolist(),
                    "arm_origin_xyz": arm_origin_xyz.astype(float).tolist(),
                }
            )

            pair["arm_id"] = int(arm_id)
            self.arm_detection_links_by_camera.setdefault(0, {})[
                int(pair["cam0_detection_index"])
            ] = int(arm_id)
            self.arm_detection_links_by_camera.setdefault(1, {})[
                int(pair["cam1_detection_index"])
            ] = int(arm_id)

            updated_ids.add(int(arm_id))
            used_existing_ids.add(int(arm_id))

        stale_arm_ids = []
        for arm_id, arm_obj in self.arm_objects.items():
            if arm_id in updated_ids:
                continue
            arm_obj["missed_frames"] = int(arm_obj.get("missed_frames", 0)) + 1
            if arm_obj["missed_frames"] > self.arm_link_max_missed_frames:
                stale_arm_ids.append(arm_id)
        for arm_id in stale_arm_ids:
            self.arm_objects.pop(arm_id, None)
            self.arm_controller_links.pop(arm_id, None)

    def _decay_arm_links(self):
        stale_arm_ids = []
        for arm_id, arm_obj in self.arm_objects.items():
            arm_obj["missed_frames"] = int(arm_obj.get("missed_frames", 0)) + 1
            if arm_obj["missed_frames"] > self.arm_link_max_missed_frames:
                stale_arm_ids.append(arm_id)
        for arm_id in stale_arm_ids:
            self.arm_objects.pop(arm_id, None)
            self.arm_controller_links.pop(arm_id, None)

    def match_arm_keypoints_between_cameras(self, max_pair_cost=25.0):
        self._arm_match_frame_index += 1
        self.arm_keypoint_matches = []
        self.arm_detection_links_by_camera = {0: {}, 1: {}}

        if 0 not in self.cameras or 1 not in self.cameras:
            self._decay_arm_links()
            return []

        camera0 = self.cameras[0]
        camera1 = self.cameras[1]
        if camera0.projMatrix is None or camera1.projMatrix is None:
            self._decay_arm_links()
            return []

        detections0 = self.arm_detections_by_camera.get(0, [])
        detections1 = self.arm_detections_by_camera.get(1, [])
        if not detections0 or not detections1:
            self._decay_arm_links()
            return []

        candidates = []
        for idx0, detection0 in enumerate(detections0):
            base0 = self._get_arm_detection_keypoint_xy(detection0, "base")
            if base0 is None:
                continue

            for idx1, detection1 in enumerate(detections1):
                base1 = self._get_arm_detection_keypoint_xy(detection1, "base")
                if base1 is None:
                    continue

                base3d, base_err = self._triangulate_arm_keypoint(camera0, camera1, base0, base1)
                if base3d is None:
                    continue
                if base_err is None:
                    base_err = 0.0

                keypoints_3d = {}
                total_error = 0.0
                num_matched_keypoints = 0

                for keypoint_name in self.arm_keypoint_names:
                    point0 = self._get_arm_detection_keypoint_xy(detection0, keypoint_name)
                    point1 = self._get_arm_detection_keypoint_xy(detection1, keypoint_name)
                    if point0 is None or point1 is None:
                        continue

                    point3d, reproj_error = self._triangulate_arm_keypoint(
                        camera0,
                        camera1,
                        point0,
                        point1,
                    )
                    if point3d is None:
                        continue

                    keypoints_3d[keypoint_name] = point3d.astype(float).tolist()
                    if reproj_error is not None:
                        total_error += float(reproj_error)
                    num_matched_keypoints += 1

                if "base" not in keypoints_3d:
                    keypoints_3d["base"] = base3d.astype(float).tolist()
                    num_matched_keypoints += 1

                if num_matched_keypoints <= 0:
                    continue

                avg_error = total_error / max(1, num_matched_keypoints)
                pair_cost = float(avg_error + 0.25 * float(base_err))
                candidates.append(
                    {
                        "cam0_detection_index": int(idx0),
                        "cam1_detection_index": int(idx1),
                        "keypoints_3d": keypoints_3d,
                        "num_keypoints": int(num_matched_keypoints),
                        "cost": pair_cost,
                    }
                )

        matched_pairs = []
        used0 = set()
        used1 = set()
        for candidate in sorted(candidates, key=lambda item: item["cost"]):
            if candidate["cost"] > max_pair_cost:
                continue
            idx0 = candidate["cam0_detection_index"]
            idx1 = candidate["cam1_detection_index"]
            if idx0 in used0 or idx1 in used1:
                continue
            used0.add(idx0)
            used1.add(idx1)
            matched_pairs.append(candidate)

        self._link_matched_arms(matched_pairs)
        self.arm_keypoint_matches = matched_pairs
        return matched_pairs

    def match_global_tracks(self):
        self.match_arm_keypoints_between_cameras()

        if 0 not in self.cameras or 1 not in self.cameras:
            self.global_tracks = {}
            return

        c0 = self.cameras[0]
        c1 = self.cameras[1]
        required = [
            getattr(c0, "projMatrix", None), getattr(c1, "projMatrix", None),
            getattr(c0, "r", None), getattr(c1, "r", None),
            getattr(c0, "t", None), getattr(c1, "t", None),
            getattr(c0, "model_plus", None), getattr(c1, "model_plus", None),
        ]
        if any(v is None for v in required):
            self.global_tracks = {}
            return
        if c0.model_plus.detections_tracked is None or c1.model_plus.detections_tracked is None:
            self.global_tracks = {}
            return

        cam0_tracks = self.cameras[0].model_plus.detections_tracked.tracker_id.tolist()
        cam1_tracks = self.cameras[1].model_plus.detections_tracked.tracker_id.tolist()

        if len(cam0_tracks) == 0 and len(cam1_tracks) ==     0:
            self.global_tracks = {}
            return
        elif len(cam0_tracks) == 0:
            self.global_tracks = {(None, tid1): None for tid1 in cam1_tracks}
            return
        elif len(cam1_tracks) == 0:
            self.global_tracks = {(tid0, None): None for tid0 in cam0_tracks}
            return

        # Build Cost Matrix with Reprojection Error
        #################################################
        xyxy = self.cameras[0].model_plus.detections_tracked.xyxy
        centers0 = [( (box[0]+box[2])/2, (box[1]+box[3])/2 ) for box in xyxy]
        class0 = self.cameras[0].model_plus.detections_tracked.class_id
        color0,shape0 = self.class2colorshape(class0)
        confidence0 = self.cameras[0].model_plus.detections_tracked.confidence
        xyxy = self.cameras[1].model_plus.detections_tracked.xyxy
        centers1 = [( (box[0]+box[2])/2, (box[1]+box[3])/2 ) for box in xyxy]
        class1 = self.cameras[1].model_plus.detections_tracked.class_id
        color1,shape1 = self.class2colorshape(class1)
        confidence1 = self.cameras[1].model_plus.detections_tracked.confidence

        
        costs = np.ones((len(cam0_tracks), len(cam1_tracks))) * 1e6
        points_3d = np.ones((len(cam0_tracks), len(cam1_tracks), 3))

        for i, tid0 in enumerate(cam0_tracks):
            for j, tid1 in enumerate(cam1_tracks):
                # Convert center points to numpy arrays
                center0 = np.array(centers0[i], dtype=np.float32)
                center1 = np.array(centers1[j], dtype=np.float32)
                
                point_3d = cv2.triangulatePoints(self.cameras[0].projMatrix,
                                        self.cameras[1].projMatrix,
                                        center0,
                                        center1)
                point_3d = point_3d / point_3d[3]
                point3d = point_3d[:3].reshape(-1, 1)

                points_3d[i, j] = point3d.flatten()

                reproject_c0 = cv2.projectPoints(point3d, self.cameras[0].r, self.cameras[0].t, self.cameras[0].mtx, self.cameras[0].dist)[0].flatten()
                reproject_c1 = cv2.projectPoints(point3d, self.cameras[1].r, self.cameras[1].t, self.cameras[1].mtx, self.cameras[1].dist)[0].flatten()
                costs[i,j] = np.linalg.norm(reproject_c0 - center0) + np.linalg.norm(reproject_c1 - center1)

                if color0[i]!=color1[j] and shape0[i]==shape1[j]:
                    costs[i,j] = 1.2*costs[i,j]
                if color0[i]==color1[j] and shape0[i]!=shape1[j]:
                    costs[i,j] = 1.1*costs[i,j]
                if color0[i]!=color1[j] and shape0[i]==shape1[j]:
                    costs[i,j] = 1.4*costs[i,j]


        # Add Rows and columns to the cost matrix for tracks without matches
        #for i in range(unmatched_0):
        #    costs= np.append(costs, np.ones((costs.shape[0], 1)) * (3 + i / 10), axis=1)
        #for i in range(unmatched_1):
        #    costs= np.append(costs, np.ones((1, costs.shape[1])) * (3 + i / 10), axis=0)
        #costs[len(unmatched_0):, len(unmatched_1):] = 1e6

        # Use Greedy Heuristic to Solve Assignment Problem
        ###########################################################

        cost = 0
        assignments = {}
        unmatched_0 = cam0_tracks.copy()
        unmatched_1 = cam1_tracks.copy()
        while cost < 15:
            min_row, min_col = np.unravel_index(np.argmin(costs), costs.shape)
            #print(f"    for tracks {cam0_tracks[min_row]} and {cam1_tracks[min_col]}")
            cost = costs[min_row, min_col]
            if cost < 15:
                tid0 = cam0_tracks[min_row]
                tid1 = cam1_tracks[min_col]
                
                # Determine the Global Track Color
                if color0[min_row] == color1[min_col]:
                    colorg = color0[min_row]
                elif confidence0[min_row] > confidence1[min_col]:
                    colorg = color0[min_row]
                else:
                    colorg = color1[min_col]

                #Determine the Global Track Shape
                if shape0[min_row] == shape1[min_col]:
                    shapeg = shape0[min_row]
                elif confidence0[min_row] > confidence1[min_col]:
                    shapeg = shape0[min_row]
                else:
                    shapeg = shape1[min_col]

                assignments[(tid0, tid1)] = [points_3d[min_row, min_col],colorg,shapeg]
                unmatched_0.remove(tid0)
                unmatched_1.remove(tid1)
                costs[min_row, :] = 1e6
                costs[:, min_col] = 1e6

        for tid in unmatched_0:
            assignments[(tid, None)] = None
        for tid in unmatched_1:
            assignments[(None, tid)] = None

        self.global_tracks = assignments

    def class2colorshape(self,class_list):
        color = []
        shape = []
        for c in class_list:
            if c in [0,2,4]:
                shape.append("square")
            else:
                shape.append("circle")

            if c in [0,1]:
                color.append("red")
            elif c in [2,3]:
                color.append("green")
            else:
                color.append("blue")
        return color,shape

    def _annotate_arm_detections(self, frame, camera_number):
        detections = self.arm_detections_by_camera.get(camera_number, [])
        arm_links = self.arm_detection_links_by_camera.get(camera_number, {})
        skeleton_edges = (
            ("base", "shoulder"),
            ("shoulder", "elbow"),
            ("elbow", "wrist"),
            ("wrist", "gripper"),
        )
        for detection_idx, detection in enumerate(detections):
            xyxy = detection.get("xyxy", [])
            if len(xyxy) != 4:
                continue
            p1 = Camera._to_cv_point((xyxy[0], xyxy[1]))
            p2 = Camera._to_cv_point((xyxy[2], xyxy[3]))
            if p1 is None or p2 is None:
                continue

            label = detection.get("label", "arm")
            conf = detection.get("confidence")
            arm_id = arm_links.get(detection_idx)
            arm_obj = self.arm_objects.get(int(arm_id)) if arm_id is not None else None
            is_calibrated = bool(arm_obj.get("calibrated", False)) if arm_obj is not None else False
            is_in_scene = bool(arm_obj.get("added_to_scene", False)) if arm_obj is not None else False
            is_matched = arm_id is not None

            if is_calibrated and is_matched:
                box_color = (0, 235, 255)
                box_thickness = 2
                keypoint_color = (0, 255, 255)
                keypoint_radius = 5
                keypoint_thickness = -1
                edge_color = (0, 220, 255)
                edge_thickness = 4
            elif is_matched and is_in_scene:
                box_color = (0, 200, 255)
                box_thickness = 2
                keypoint_color = (0, 220, 255)
                keypoint_radius = 5
                keypoint_thickness = 1
                edge_color = (0, 200, 255)
                edge_thickness = 2
            else:
                box_color = (120, 180, 230)
                box_thickness = 1
                keypoint_color = (120, 220, 255)
                keypoint_radius = 4
                keypoint_thickness = 1
                edge_color = None
                edge_thickness = 0

            cv2.rectangle(frame, p1, p2, box_color, box_thickness)

            if conf is None:
                text = f"arm:{label}"
            else:
                text = f"arm:{label} {float(conf):.2f}"
            if arm_id is not None:
                text = f"{text} id:{int(arm_id)}"
            if is_calibrated:
                text = f"{text} cal"
            elif is_in_scene:
                text = f"{text} scene"
            text_origin = (int(p1[0]), int(max(14, p1[1] - 6)))
            cv2.putText(
                frame,
                text,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

            keypoint_pixels = {}
            for keypoint_name in self.arm_keypoint_names:
                keypoint = detection.get("keypoints", {}).get(keypoint_name)
                if keypoint is None:
                    continue
                keypoint_px = Camera._to_cv_point(keypoint.get("xy", []))
                if keypoint_px is None:
                    continue
                keypoint_pixels[keypoint_name] = keypoint_px

            if edge_color is not None:
                for edge_a, edge_b in skeleton_edges:
                    pt_a = keypoint_pixels.get(edge_a)
                    pt_b = keypoint_pixels.get(edge_b)
                    if pt_a is None or pt_b is None:
                        continue
                    cv2.line(frame, pt_a, pt_b, edge_color, edge_thickness)

            for keypoint_name in self.arm_keypoint_names:
                keypoint_px = keypoint_pixels.get(keypoint_name)
                if keypoint_px is None:
                    continue
                cv2.circle(frame, keypoint_px, keypoint_radius, keypoint_color, keypoint_thickness)
                cv2.putText(
                    frame,
                    keypoint_name[0].upper(),
                    (keypoint_px[0] + 4, keypoint_px[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    keypoint_color,
                    1,
                )

    def annotated_frame_from_camera(self, camera_number):
        af = self.cameras[camera_number].annotate_frame() # This triggers the model_plus and camera to annotate the frame
        if af is None:
            return None

        # Draw Global Track Positions
        for match,values in self.global_tracks.items():
            if values is None:
                continue
            point_3d,color,shape = values
            if match[0] is not None and match[1] is not None:
                point_3d_reshaped = point_3d.reshape(-1, 1)
                cam0_point = cv2.projectPoints(point_3d_reshaped, self.cameras[camera_number].r, self.cameras[camera_number].t, self.cameras[camera_number].mtx, self.cameras[camera_number].dist)[0].flatten()
                point_px = self.cameras[camera_number]._to_cv_point(cam0_point[:2])
                if point_px is None:
                    continue
                cv2.circle(af, point_px, 5, (0,0,255), -1)
                p3d = point_3d.flatten()
                label_px = (point_px[0], point_px[1] - 10)
                cv2.putText(
                    af,
                    f"{p3d[0]:.1f}, {p3d[1]:.1f}, {p3d[2]:.1f}",
                    label_px,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,0,255),
                    2,
                )

        self._annotate_arm_detections(af, camera_number)
        self._annotate_arm_origin_overlays(af, camera_number)
        return af

    def plot_overhead(self, save_path=None):
            """Plot all cameras' FOVs and all global detections in an overhead view."""
            fig, ax = plt.subplots(figsize=(6,6))
            ax.set_aspect('equal')
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)")
            ax.set_title("Overhead View of Multi-Camera Multi-Object Tracker")

            # Plot global track detections
            if self.global_tracks:
                first = True
                for match,values in self.global_tracks.items():
                    if values is None:
                        continue
                    point,color,shape = values
                    p = np.array(point).flatten()
                    if shape == "square":
                        marker_type = "s"
                    else:
                        marker_type = "o"

                    if p.size >= 2:  # make sure it has x,y
                        ax.scatter(p[0], p[1], s=70, c=color, marker=marker_type,label='Detections' if first else "")
                        first = False

            # Plot arm base markers and calibrated arm axes.
            arm_base_first = True
            for arm_id in sorted(self.arm_objects.keys()):
                arm_obj = self.arm_objects.get(arm_id, {})
                if not bool(arm_obj.get("added_to_scene", False)):
                    continue
                base_xyz = self._get_arm_base_marker_xyz(arm_obj)
                if base_xyz is not None:
                    ax.scatter(
                        float(base_xyz[0]),
                        float(base_xyz[1]),
                        s=110,
                        c="orange",
                        marker="x",
                        linewidths=2.2,
                        label="Arm Base" if arm_base_first else "",
                    )
                    ax.text(
                        float(base_xyz[0]),
                        float(base_xyz[1]),
                        f"Arm {int(arm_id)} ({base_xyz[0]:.1f}, {base_xyz[1]:.1f}, {base_xyz[2]:.1f})",
                        color="darkorange",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )
                    arm_base_first = False

                if not bool(arm_obj.get("calibrated", False)):
                    continue

                axis_origin = self._normalize_xyz(arm_obj.get("arm_origin_xyz"))
                if axis_origin is None:
                    axis_origin = base_xyz
                if axis_origin is None:
                    continue

                rotation = self._get_arm_rotation_robot_to_global(arm_id)
                if rotation is None:
                    rotation = np.eye(3, dtype=np.float64)
                axis_names = ("x", "y", "z")
                for axis_idx, axis_name in enumerate(axis_names):
                    tip = axis_origin + float(self.arm_axis_length) * np.asarray(
                        rotation[:, axis_idx],
                        dtype=np.float64,
                    )
                    ax.annotate(
                        "",
                        xy=(float(tip[0]), float(tip[1])),
                        xytext=(float(axis_origin[0]), float(axis_origin[1])),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": "darkorange",
                            "lw": 1.8,
                        },
                    )
                    ax.text(
                        float(tip[0]),
                        float(tip[1]),
                        axis_name,
                        color="darkorange",
                        fontsize=9,
                        fontweight="bold",
                    )

            # Plot each camera's FOV
            for cam_id, cam in self.cameras.items():
                fov = cam.compute_fov_on_ground()
                if not fov:
                    continue
                C, left, right = fov["C"], fov["left"], fov["right"]

                ax.fill(
                    [C[0], left[0], right[0]],
                    [C[1], left[1], right[1]],
                    color="royalblue",
                    alpha=0.08,
                )
                ax.plot([C[0], left[0]], [C[1], left[1]], "b--", linewidth=1.2)
                ax.plot([C[0], right[0]], [C[1], right[1]], "b--", linewidth=1.2)
                ax.scatter(C[0], C[1], c="royalblue", marker="^", s=70)
                ax.text(C[0], C[1], f"Cam {cam_id}", color="royalblue", fontsize=8, ha="right")

            # Convert Matplotlib figure to OpenCV image
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            self.overhead_plot = img
            return self.overhead_plot 

    def _ensure_open3d(self):
        if self._open3d_available is True:
            return self._open3d_module
        if self._open3d_available is False:
            return None

        try:
            import open3d as o3d
        except ImportError:
            self._open3d_available = False
            return None

        self._open3d_module = o3d
        self._open3d_available = True
        return o3d

    def _get_stereo_cameras(self):
        if 0 not in self.cameras or 1 not in self.cameras:
            raise RuntimeError("Scene mesh display requires two calibrated cameras (camera ids 0 and 1).")
        return self.cameras[0], self.cameras[1]

    def _compute_relative_transform_from_world_poses(self):
        cam0, cam1 = self._get_stereo_cameras()
        required = {
            "cam0.r": getattr(cam0, "r", None),
            "cam0.tvec": getattr(cam0, "tvec", None),
            "cam1.r": getattr(cam1, "r", None),
            "cam1.tvec": getattr(cam1, "tvec", None),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                "Cannot compute stereo transform before world calibration. Missing: "
                + ", ".join(missing)
            )

        R0 = np.asarray(cam0.r, dtype=np.float64)
        t0 = np.asarray(cam0.tvec, dtype=np.float64).reshape(3, 1)
        R1 = np.asarray(cam1.r, dtype=np.float64)
        t1 = np.asarray(cam1.tvec, dtype=np.float64).reshape(3, 1)

        # world->camera poses to camera0->camera1 transform
        R_rel = R1 @ R0.T
        T_rel = t1 - R_rel @ t0
        return R_rel, T_rel

    def _ensure_stereo_rectification(self):
        cam0, cam1 = self._get_stereo_cameras()
        if cam0.frame is None or cam1.frame is None:
            self.capture_cameras_frames()
        if cam0.frame is None or cam1.frame is None:
            raise RuntimeError("Unable to capture frames for stereo rectification.")

        h0, w0 = cam0.frame.shape[:2]
        h1, w1 = cam1.frame.shape[:2]
        if (w0, h0) != (w1, h1):
            raise RuntimeError(
                "Stereo mesh display requires same frame size from both cameras. "
                f"Got cam0 {(w0, h0)} vs cam1 {(w1, h1)}."
            )

        image_size = (w0, h0)
        if self.mesh_stereo_ready and self.mesh_rect_size == image_size:
            return

        if cam0.mtx is None or cam0.dist is None or cam1.mtx is None or cam1.dist is None:
            raise RuntimeError("Missing camera intrinsics; run camera calibration first.")

        R_rel, T_rel = self._compute_relative_transform_from_world_poses()
        R1_rect, R2_rect, P1_rect, P2_rect, Q, _, _ = cv2.stereoRectify(
            cameraMatrix1=cam0.mtx,
            distCoeffs1=cam0.dist,
            cameraMatrix2=cam1.mtx,
            distCoeffs2=cam1.dist,
            imageSize=image_size,
            R=R_rel,
            T=T_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        self.mesh_map0x, self.mesh_map0y = cv2.initUndistortRectifyMap(
            cam0.mtx, cam0.dist, R1_rect, P1_rect, image_size, cv2.CV_32FC1
        )
        self.mesh_map1x, self.mesh_map1y = cv2.initUndistortRectifyMap(
            cam1.mtx, cam1.dist, R2_rect, P2_rect, image_size, cv2.CV_32FC1
        )
        self.mesh_Q = Q
        self.mesh_R_rel = R_rel
        self.mesh_T_rel = T_rel
        self.mesh_rect_size = image_size
        self.mesh_stereo_ready = True

    def _compute_dense_point_cloud(self):
        if self.mesh_stereo_matcher is None:
            raise RuntimeError("Scene mesh stereo matcher is disabled.")

        self._ensure_stereo_rectification()
        cam0, cam1 = self._get_stereo_cameras()
        if cam0.frame is None or cam1.frame is None:
            self.capture_cameras_frames()
        if cam0.frame is None or cam1.frame is None:
            raise RuntimeError("Unable to capture frames for dense depth.")

        rect0 = cv2.remap(cam0.frame, self.mesh_map0x, self.mesh_map0y, cv2.INTER_LINEAR)
        rect1 = cv2.remap(cam1.frame, self.mesh_map1x, self.mesh_map1y, cv2.INTER_LINEAR)
        gray0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)

        disparity = self.mesh_stereo_matcher.compute(gray0, gray1).astype(np.float32) / 16.0
        points_3d = cv2.reprojectImageTo3D(disparity, self.mesh_Q)
        colors_rgb = cv2.cvtColor(rect0, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        valid = (
            np.isfinite(points_3d).all(axis=2)
            & (disparity > self.mesh_depth_min_disparity)
            & (np.abs(points_3d[:, :, 2]) < self.mesh_depth_max)
        )

        points = points_3d[valid]
        colors = colors_rgb[valid]
        if points.size == 0:
            raise RuntimeError("Dense stereo produced no valid 3D points.")

        if len(points) > self.mesh_max_points:
            keep = np.random.choice(len(points), size=self.mesh_max_points, replace=False)
            points = points[keep]
            colors = colors[keep]

        return points.astype(np.float64), colors.astype(np.float64)

    def _sample_vertex_colors(self, mesh, pcd, o3d):
        pcd_colors = np.asarray(pcd.colors)
        if len(pcd_colors) == 0 or len(mesh.vertices) == 0:
            return

        tree = o3d.geometry.KDTreeFlann(pcd)
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
        vertices = np.asarray(mesh.vertices)
        for i, v in enumerate(vertices):
            _, idx, _ = tree.search_knn_vector_3d(v, 1)
            if idx:
                vertex_colors[i] = pcd_colors[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    def _build_scene_mesh(self, points, colors):
        o3d = self._ensure_open3d()
        if o3d is None:
            raise RuntimeError("open3d is required for mesh reconstruction.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        scene_extent = np.ptp(points, axis=0)
        scene_scale = float(np.linalg.norm(scene_extent))
        voxel_size = max(scene_scale / 300.0, 1e-3)
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        if len(pcd.points) < 200:
            raise RuntimeError("Too few points after downsampling for mesh reconstruction.")

        normal_radius = max(scene_scale / 80.0, voxel_size * 2.0, 1e-3)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
        )

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.mesh_poisson_depth
        )
        densities = np.asarray(densities)
        if densities.size > 0:
            density_threshold = np.quantile(densities, 0.05)
            mesh.remove_vertices_by_mask(densities < density_threshold)

        if len(mesh.triangles) > self.mesh_triangle_budget:
            mesh = mesh.simplify_quadric_decimation(self.mesh_triangle_budget)
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        mesh.compute_vertex_normals()

        # Approximate texture by transferring nearest point colors to mesh vertices.
        self._sample_vertex_colors(mesh, pcd, o3d)
        return mesh

    def _export_scene_mesh(self, mesh, export_path=None):
        if export_path is None:
            export_path = self.mesh_export_path
        if not export_path:
            return

        o3d = self._ensure_open3d()
        out_path = Path(export_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_colors=True)
        except Exception:
            ok = False
        if ok:
            self.mesh_status = f"Mesh exported: {out_path}"
            return

        fallback = out_path.with_suffix(".ply")
        fallback_ok = o3d.io.write_triangle_mesh(str(fallback), mesh, write_vertex_colors=True)
        if fallback_ok:
            self.mesh_status = (
                f"Mesh export to {out_path.name} failed; exported fallback mesh: {fallback}"
            )
        else:
            self.mesh_status = f"Mesh export failed: {out_path}"

    def _display_scene_mesh(self, mesh):
        o3d = self._ensure_open3d()
        if self.mesh_visualizer is None:
            self.mesh_visualizer = o3d.visualization.Visualizer()
            self.mesh_visualizer.create_window(self.mesh_window_name, width=960, height=720)

        self.mesh_visualizer.clear_geometries()
        self.mesh_visualizer.add_geometry(mesh)
        self.mesh_visualizer.poll_events()
        self.mesh_visualizer.update_renderer()

    def update_scene_mesh_display(self, force=False):
        if not self.mesh_display:
            return

        o3d = self._ensure_open3d()
        if o3d is None:
            if not self._warned_open3d_missing:
                print("Scene mesh display disabled: install open3d to enable mesh rendering/export.")
                self._warned_open3d_missing = True
            return

        now = time.time()
        if not force and now - self.mesh_last_update_time < self.mesh_update_interval_sec:
            if self.mesh_visualizer is not None:
                self.mesh_visualizer.poll_events()
                self.mesh_visualizer.update_renderer()
            return

        try:
            points, colors = self._compute_dense_point_cloud()
            mesh = self._build_scene_mesh(points, colors)
            self.scene_mesh = mesh
            self._display_scene_mesh(mesh)
            self._export_scene_mesh(mesh)
            self.mesh_last_update_time = now
        except Exception as exc:
            self.mesh_status = f"Scene mesh update failed: {exc}"
            print(self.mesh_status)
    
    def update_displays(self):
        for camera in self.cameras.values():
            camera.annotate_frame()
            camera.display_frame()

        self.plot_overhead()
        if self.gui_available:
            cv2.imshow("Overhead View", self.overhead_plot)
        if self.mesh_display:
            self.update_scene_mesh_display()
        if self.gui_available:
            cv2.waitKey(1)

'''
        # Rematch old matches first
        #####################################################################3
        for mct in self.tracks:

            if len(mct.last_camera_match) < 2:
                continue


            elif len(mct.last_camera_match) == 2:
                prev_location = self.cameras[cids[0]].tracks[tid0].center
                tid0 = mct.last_camera_match[cids[0]]
                tid1 = mct.last_camera_match[cids[1]]

                points_3d = cv2.triangulatePoints(self.cameras[cids[0]].projMatrix,
                                        self.cameras[cids[1]].projMatrix,
                                        self.cameras[cids[0]].tracks[tid0].center,
                                        self.cameras[cids[1]].tracks[tid1].center)
                # If new point is withing .2 of prev, match
                if np.linalg.norm(points_3d - prev_location) < 0.2:
                    mct.set_match([self.cameras[cids[0]].tracks[tid0], self.cameras[cids[1]].tracks[tid1]])
                    self.cameras[cids[0]].tracks[tid0].global_id = mct.global_id
                    self.cameras[cids[1]].tracks[tid1].global_id = mct.global_id


                    unmatched_0.remove(tid0)
                    unmatched_1.remove(tid1)
                    continue
                    


    # Solve the assignment problem

        # Solve Assignment with Google OR Tools SAT Solver
        ###########################

        # Create the model
        model = cp_model.CpModel()

        # Create assignment variables, x[w, t] is 1 if worker w is assigned to task t.
        x = {}
        for w in range(costs.shape[0]):
            for t in range(costs.shape[1]):
                x[w, t] = model.NewBoolVar(f'x_{w}_{t}')

        # Add constraints: Each worker is assigned at most one task.
        for w in range(len(unmatched_0)):
            model.Add(sum(x[w, t] for t in range(costs.shape[1])) <= 1)

        # Add constraints: Each task is assigned to exactly one worker.
        for t in range(len(unmatched_1)):
            model.Add(sum(x[w, t] for w in range(len(unmatched_0))) == 1)

        # Create the objective function.
        objective_terms = []
        for w in range(len(unmatched_0)):
            for t in range(len(unmatched_1)):
                objective_terms.append(costs[w][t] * x[w, t])
        model.Minimize(sum(objective_terms))

        # Create the solver and solve the model.
        solver = cp_model.CpSolver()

        # --- Provide the initial guess (warm start) ---
        single_ids_0 = 0
        single_ids_1 = 0
        initial_assignments = {}
        for gt in self.global_tracks:
            # Check if tracks are in recent update
            if gt.last_camera_match[0] not in unmatched_0:
                gt.last_camera_match[0] = None
            if gt.last_camera_match[1] not in unmatched_1:
                gt.last_camera_match[1] = None

            if gt.last_camera_match[0] is None and gt.last_camera_match[1] is None:
                print("Error: Global track with no camera matches")

            elif gt.last_camera_match[0] is not None and gt.last_camera_match[1] is not None:
                tid0 = gt.last_camera_match[0]
                tid1 = gt.last_camera_match[1]
                row_in_costs = unmatched_0.index(tid0)
                col_in_costs = unmatched_1.index(tid1)
                initial_assignments[(row_in_costs, col_in_costs)] = 1

            elif gt.last_camera_match[0] is not None and gt.last_camera_match[1] is None:
                cid = gt.last_camera_match.keys()[0]
                tid = gt.last_camera_match[cid]
                row_in_costs = unmatched_0.index(tid)
                col_initial_assignment = len(unmatched_1) + single_ids_0
                initial_assignments[(row_in_costs, col_initial_assignment)] = 1
                single_ids_0 += 1
            elif gt.last_camera_match[0] is None and gt.last_camera_match[1] is not None:
                row_initial_assignment = len(unmatched_0) + single_ids_1
                col_initial_assignment = unmatched_1.index(tid)
                initial_assignments[(row_initial_assignment, col_initial_assignment)] = 1
                single_ids_1 += 1
            else:
                print("Global Track Could Not be Resolved To an Inital Guess for Assignement Problem")

        # Apply the initial guess to the solver.
        for (w, t), val in initial_assignments.items():
            solver.AddHint(x[w, t], val)

        # Solve the problem.
        status = solver.Solve(model)

        # Print the solution.
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"Total cost: {solver.ObjectiveValue()}")
            for w in range(num_workers):
                for t in range(num_tasks):
                    if solver.BooleanValue(x[w, t]):
                        print(f"Worker {w} assigned to task {t}. Cost: {costs[w][t]}")
        else:
            print("No solution found.")



'''
