from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .Camera import Camera, detector
from .MCMOTracker import MCMOTracker
from .config.ArucoConfig import ArucoConfig


class MCMOTService(MCMOTracker):
    """
    Step-by-step MCMOT orchestration class intended for UI-driven workflows.

    Lifecycle:
    1) instantiate -> discover available cameras/models, open first 2 cameras as raw preview
    2) select cameras -> change camera ports
    3) attach parts model -> parts detections appear on frames
    4) run calibration explicitly (auto / aruco / landmarks)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        tracker_yaml_path: str = "./config/bytetrack.yaml",
        default_camera_name: str = "logitech_1",
        max_camera_search: int = 8,
        models_root: Optional[str] = None,
    ):
        super().__init__(
            model_path=None,
            camera_device_ids=[],
            camera_names=[],
            confidence_threshold=confidence_threshold,
            tracker_yaml_path=tracker_yaml_path,
            aruco_positions=None,
            display=False,
            calibration_method=None,
            mesh_display=False,
        )
        self.default_camera_name = default_camera_name
        self.max_camera_search = int(max_camera_search)
        package_root = Path(__file__).resolve().parent
        self.models_root = Path(models_root) if models_root else (package_root / "models")
        self.parts_models_root = self.models_root / "parts"
        self.arm_models_root = self.models_root / "arm"

        self.available_camera_ids: list[int] = []
        self.available_parts_models: list[str] = []
        self.available_arm_models: list[str] = []
        self.available_models: list[str] = []  # Backward-compatible alias to parts models.
        self.selected_camera_ids: list[int] = []
        self.selected_camera_names: list[str] = []
        self.parts_model_loaded: bool = False
        self.model_loaded: bool = False  # Backward-compatible alias.
        self.calibration_preview_mode: Optional[str] = None
        self.aruco_positions = "square4"
        self.aruco_config = ArucoConfig(self.aruco_positions)

        # camera_number -> {"image_points":[[x,y]], "object_points":[[x,y,z]]}
        self._landmark_state: dict[int, dict[str, list[list[float]]]] = {}

        self.refresh_options()
        default_ids = self.available_camera_ids[:2]
        if default_ids:
            self.set_active_cameras(default_ids)

    def refresh_options(self):
        self.available_camera_ids = self._discover_camera_ids(self.max_camera_search)
        self.available_parts_models = self._discover_models(self.parts_models_root)
        self.available_arm_models = self._discover_models(self.arm_models_root)
        self.available_models = list(self.available_parts_models)
        return {
            "available_camera_ids": list(self.available_camera_ids),
            "available_parts_models": list(self.available_parts_models),
            "available_arm_models": list(self.available_arm_models),
            "available_models": list(self.available_models),
        }

    @staticmethod
    def _discover_camera_ids(max_search: int = 8) -> list[int]:
        ids: list[int] = []
        for device_id in range(max_search):
            cap = cv2.VideoCapture(device_id)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if ok and frame is not None:
                    ids.append(device_id)
            finally:
                cap.release()
        return ids

    @staticmethod
    def _discover_models(models_root: Path) -> list[str]:
        if not models_root.exists():
            return []
        files = sorted(
            [
                p
                for p in models_root.rglob("*")
                if p.is_file() and p.suffix.lower() in {".pt", ".onnx", ".engine"}
            ]
        )
        return [str(p) for p in files]

    def _release_cameras(self):
        for camera in self.cameras.values():
            cap = getattr(camera, "cap", None)
            if cap is not None:
                cap.release()
        self.cameras = {}
        self.global_tracks = {}
        self._landmark_state = {}

    def set_active_cameras(self, camera_device_ids: list[int], camera_names: Optional[list[str]] = None):
        if not camera_device_ids:
            raise ValueError("camera_device_ids cannot be empty.")
        if camera_names is None:
            camera_names = [self.default_camera_name] * len(camera_device_ids)
        if len(camera_names) != len(camera_device_ids):
            raise ValueError("camera_names must match camera_device_ids length.")

        self._release_cameras()
        failures = []
        for camera_number, (device_id, camera_name) in enumerate(zip(camera_device_ids, camera_names)):
            try:
                self.cameras[camera_number] = Camera(
                    camera_number=camera_number,
                    camera_device_id=device_id,
                    camera_name=camera_name,
                    model_path=None,
                    confidence_threshold=self.parts_confidence_threshold,
                    tracker_yaml_path=self.tracker_yaml_path,
                    aruco_positions=None,
                    display=False,
                    calibration_method=None,
                    load_intrinsics=False,
                    show_world_axis=self.show_world_axis,
                    world_axis_length=self.world_axis_length,
                )
            except Exception as exc:
                failures.append((camera_number, device_id, str(exc)))

        if failures:
            self._release_cameras()
            detail = "; ".join(
                [f"cam#{num} id={device_id}: {msg}" for num, device_id, msg in failures]
            )
            raise RuntimeError(f"Failed to open selected cameras: {detail}")

        self.camera_device_ids = list(camera_device_ids)
        self.camera_names = list(camera_names)
        self.selected_camera_ids = list(camera_device_ids)
        self.selected_camera_names = list(camera_names)
        self.calibration_method = None
        self.calibration_preview_mode = None

        if self.parts_model_loaded and self.parts_model is not None and self.parts_model_path:
            self._reset_camera_trackers()
        else:
            self.parts_model_loaded = False
            self.model_loaded = False

        self.capture_cameras_frames()
        return {
            "selected_camera_ids": list(self.selected_camera_ids),
            "selected_camera_names": list(self.selected_camera_names),
        }

    def close(self):
        self._release_cameras()
        cv2.destroyAllWindows()

    def set_parts_model(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        tracker_yaml_path: Optional[str] = None,
    ):
        if not self.cameras:
            raise RuntimeError("No cameras are active. Call set_active_cameras first.")
        data = super().set_parts_model(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            tracker_yaml_path=tracker_yaml_path,
        )
        self.parts_model_loaded = True
        self.model_loaded = True
        return data

    def set_parts_confidence_threshold(self, confidence_threshold: float):
        return super().set_parts_confidence_threshold(confidence_threshold)

    def clear_parts_model(self):
        data = super().clear_parts_model()
        self.parts_model_loaded = False
        self.model_loaded = False
        return data

    def set_arm_model(self, model_path: str, confidence_threshold: Optional[float] = None):
        if not self.cameras:
            raise RuntimeError("No cameras are active. Call set_active_cameras first.")
        return super().set_arm_model(model_path, confidence_threshold=confidence_threshold)

    def clear_arm_model(self):
        return super().clear_arm_model()

    # Backward-compatible wrappers.
    def set_detection_model(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        tracker_yaml_path: Optional[str] = None,
    ):
        return self.set_parts_model(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            tracker_yaml_path=tracker_yaml_path,
        )

    def set_confidence_threshold(self, confidence_threshold: float):
        return self.set_parts_confidence_threshold(confidence_threshold)

    def clear_detection_model(self):
        return self.clear_parts_model()

    def _ensure_intrinsics_loaded(self, camera_numbers: Optional[list[int]] = None):
        if camera_numbers is None:
            camera_numbers = sorted(self.cameras.keys())
        for camera_number in camera_numbers:
            camera = self.cameras[camera_number]
            if camera.mtx is None or camera.dist is None:
                camera.set_intrinsics()

    def calibrate_auto_stepwise(self, baseline: float = 1.0):
        if len(self.cameras) < 2:
            raise RuntimeError("Auto calibration requires at least two active cameras.")
        self._ensure_intrinsics_loaded()
        self.calibration_method = "auto"
        super().calibrate_auto(baseline=baseline)
        return {
            "calibration_method": self.calibration_method,
            "camera_status": {
                str(cid): bool(cam.world_calibration_status)
                for cid, cam in self.cameras.items()
            },
        }

    def set_calibration_preview_mode(self, mode: Optional[str], aruco_positions: str = "square4"):
        if mode is not None:
            mode = str(mode).lower()
        if mode not in {None, "aruco", "landmarks"}:
            raise ValueError("preview mode must be one of: None, 'aruco', 'landmarks'")

        self.calibration_preview_mode = mode
        if mode == "aruco":
            self.aruco_positions = aruco_positions
            self.aruco_config = ArucoConfig(aruco_positions)
        return {"calibration_preview_mode": self.calibration_preview_mode}

    def calibrate_aruco_stepwise(
        self,
        aruco_positions: str = "square4",
        sample_frames: int = 10,
        sample_interval_sec: float = 0.05,
    ):
        if not self.cameras:
            raise RuntimeError("No active cameras for ArUco calibration.")
        self._ensure_intrinsics_loaded()
        self.aruco_positions = aruco_positions
        self.aruco_config = ArucoConfig(aruco_positions)
        self.calibration_method = "aruco"

        results = {}
        for camera_number, camera in self.cameras.items():
            detections: dict[int, list[np.ndarray]] = {}
            for _ in range(max(1, int(sample_frames))):
                camera.capture_frame()
                frame = camera.frame
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if ids is not None:
                    for marker_id, corner in zip(ids.flatten(), corners):
                        marker_id = int(marker_id)
                        if marker_id not in self.aruco_config.marker_positions:
                            continue
                        detections.setdefault(marker_id, []).append(corner)
                if sample_interval_sec > 0:
                    time.sleep(sample_interval_sec)

            object_points = []
            image_points = []
            for marker_id, corners_list in detections.items():
                mean_corner = np.mean(np.array(corners_list), axis=0)
                object_points.append(self.aruco_config.marker_positions[marker_id])
                image_points.append(np.mean(mean_corner[0], axis=0))

            if len(object_points) < 4:
                results[str(camera_number)] = {
                    "ok": False,
                    "reason": "too_few_markers",
                    "num_markers": len(object_points),
                }
                continue

            object_points_np = np.array(object_points, dtype=np.float32)
            image_points_np = np.array(image_points, dtype=np.float32)
            solved, rvec, tvec = cv2.solvePnP(
                object_points_np,
                image_points_np,
                camera.mtx,
                camera.dist,
            )
            if not solved:
                results[str(camera_number)] = {
                    "ok": False,
                    "reason": "solvepnp_failed",
                    "num_markers": len(object_points),
                }
                continue

            camera.set_extrinsics_from_rt(cv2.Rodrigues(rvec)[0], tvec, world_calibrated=True)
            results[str(camera_number)] = {
                "ok": True,
                "num_markers": len(object_points),
            }

        return {
            "calibration_method": self.calibration_method,
            "results": results,
        }

    def clear_landmarks(self, camera_number: Optional[int] = None):
        if camera_number is None:
            self._landmark_state = {}
            return
        self._landmark_state[camera_number] = {"image_points": [], "object_points": []}

    def add_landmark_observation(self, camera_number: int, pixel_xy, world_xyz):
        if camera_number not in self.cameras:
            raise ValueError(f"Unknown camera_number {camera_number}.")
        if len(pixel_xy) != 2 or len(world_xyz) != 3:
            raise ValueError("pixel_xy must have 2 values and world_xyz must have 3 values.")

        state = self._landmark_state.setdefault(
            camera_number, {"image_points": [], "object_points": []}
        )
        state["image_points"].append([float(pixel_xy[0]), float(pixel_xy[1])])
        state["object_points"].append([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])])

        return {
            "camera_number": camera_number,
            "num_points": len(state["image_points"]),
        }

    def solve_landmark_calibration(self, camera_number: int):
        if camera_number not in self.cameras:
            raise ValueError(f"Unknown camera_number {camera_number}.")
        self._ensure_intrinsics_loaded([camera_number])

        state = self._landmark_state.get(camera_number, {"image_points": [], "object_points": []})
        image_points = np.array(state["image_points"], dtype=np.float32)
        object_points = np.array(state["object_points"], dtype=np.float32)
        if len(image_points) < 4 or len(object_points) < 4:
            raise RuntimeError("Need at least 4 landmarks to solve calibration.")

        camera = self.cameras[camera_number]
        solved, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera.mtx,
            camera.dist,
            reprojectionError=4.0,
            iterationsCount=1000,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            raise RuntimeError(f"solvePnPRansac failed for camera {camera_number}.")
        if inliers is None or len(inliers) < 4:
            raise RuntimeError(
                f"solvePnPRansac produced too few inliers for camera {camera_number}."
            )

        camera.set_extrinsics_from_rt(cv2.Rodrigues(rvec)[0], tvec, world_calibrated=True)
        self.calibration_method = "landmarks"
        return {
            "camera_number": camera_number,
            "inliers": int(len(inliers)),
            "num_points": int(len(image_points)),
        }

    def get_landmark_state(self, camera_number: int):
        state = self._landmark_state.get(camera_number, {"image_points": [], "object_points": []})
        return {
            "camera_number": camera_number,
            "image_points": list(state["image_points"]),
            "object_points": list(state["object_points"]),
        }

    def annotated_frame_from_camera(self, camera_number):
        camera = self.cameras.get(camera_number)
        if camera is None:
            return None

        has_scene_arms = any(
            bool(arm_obj.get("added_to_scene", False))
            for arm_obj in self.arm_objects.values()
        )
        camera_has_axis = bool(getattr(camera, "show_world_axis", False)) and bool(
            camera._can_draw_world_axis()
        )

        # In preview mode, keep raw frames by default, but switch to annotated
        # rendering when we have models, calibrated camera axes, or scene-arm overlays.
        if self.parts_model_loaded or self.arm_model_loaded or camera_has_axis or has_scene_arms:
            frame = super().annotated_frame_from_camera(camera_number)
        else:
            frame = camera.frame.copy() if camera.frame is not None else None

        if frame is None:
            return None

        if self.calibration_preview_mode == "aruco":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for i, corner in enumerate(corners):
                    c = corner[0]
                    center = Camera._to_cv_point(c.mean(axis=0))
                    if center is not None:
                        cv2.putText(
                            frame,
                            f"ID:{ids[i][0]}",
                            center,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

        if self.calibration_preview_mode == "landmarks":
            state = self._landmark_state.get(camera_number, {"image_points": []})
            for idx, pt in enumerate(state["image_points"]):
                px = (int(pt[0]), int(pt[1]))
                cv2.circle(frame, px, 5, (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"L{idx + 1}",
                    (px[0] + 8, px[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        return frame
