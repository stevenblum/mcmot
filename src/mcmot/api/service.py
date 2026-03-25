from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from mcmot.MCMOTService import MCMOTService

from .schemas import SessionConfig, SessionStatus


@dataclass
class _RuntimeState:
    running: bool = False
    started_at: Optional[float] = None
    fps: float = 0.0
    last_loop_ms: float = 0.0
    last_error: Optional[str] = None
    calibration_method: Optional[str] = None
    mesh_display: bool = False
    available_streams: list[str] = field(default_factory=list)
    tracks_3d: list[dict] = field(default_factory=list)
    detections_per_camera: dict[str, int] = field(default_factory=dict)
    available_camera_ids: list[int] = field(default_factory=list)
    selected_camera_ids: list[int] = field(default_factory=list)
    selected_camera_names: list[str] = field(default_factory=list)
    available_parts_models: list[str] = field(default_factory=list)
    available_arm_models: list[str] = field(default_factory=list)
    selected_parts_model_path: Optional[str] = None
    parts_model_loaded: bool = False
    parts_confidence_threshold: float = 0.25
    parts_nms_enabled: bool = True
    parts_nms_iou_threshold: float = 0.7
    selected_arm_model_path: Optional[str] = None
    arm_model_loaded: bool = False
    arm_confidence_threshold: float = 0.25
    arm_nms_enabled: bool = True
    arm_nms_iou_threshold: float = 0.4
    arm_inference_interval_sec: float = 0.5
    arm_connected: bool = False
    arm_linked_id: Optional[int] = None
    arm_linked_no_control: Optional[bool] = None
    available_models: list[str] = field(default_factory=list)
    selected_model_path: Optional[str] = None
    model_loaded: bool = False
    confidence_threshold: float = 0.25
    overhead_update_interval_sec: float = 2.0
    calibration_preview_mode: Optional[str] = None
    click_mode: str = "none"
    available_click_modes: list[str] = field(default_factory=list)
    last_click: Optional[dict] = None
    pending_landmark_pixels: dict[str, list[float]] = field(default_factory=dict)
    pending_arm_base_selections: dict[str, dict[str, Any]] = field(default_factory=dict)
    config: Optional[dict] = None


class MCMOTBackgroundService:
    """
    Preview-first background service.

    - Starts immediately in preview mode (raw camera frames, no model).
    - start(config) attaches parts model (and optionally arm model / selected cameras).
    - stop() detaches models and returns to preview mode.
    """

    def __init__(self, max_camera_search: int = 8) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._loop_delay_sec = 0.0
        self._jpeg_quality = 80
        self._display_overhead = True
        self._last_overhead_ts = 0.0
        self._last_overhead_jpg: Optional[bytes] = None
        self._pending_landmark_pixels: dict[int, list[float]] = {}
        self._pending_arm_base_selections: dict[int, dict[str, Any]] = {}
        self._click_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._click_mode = "none"
        self.register_click_handler("none", self._handle_click_none)
        self.register_click_handler("landmark_select", self._handle_click_landmark_select)
        self.register_click_handler("arm_base_select", self._handle_click_arm_base_select)

        self._mcmot = MCMOTService(max_camera_search=max_camera_search)
        linked_arm_id = (
            sorted(self._mcmot.arm_controller_links.keys())[0]
            if getattr(self._mcmot, "arm_controller_links", None)
            else None
        )
        linked_arm_obj = (
            self._mcmot.arm_objects.get(linked_arm_id)
            if linked_arm_id is not None
            else None
        )
        self._runtime = _RuntimeState(
            running=True,
            started_at=time.time(),
            calibration_method=self._mcmot.calibration_method,
            mesh_display=bool(self._mcmot.mesh_display),
            available_camera_ids=list(self._mcmot.available_camera_ids),
            selected_camera_ids=list(self._mcmot.selected_camera_ids),
            selected_camera_names=list(self._mcmot.selected_camera_names),
            available_parts_models=list(self._mcmot.available_parts_models),
            available_arm_models=list(self._mcmot.available_arm_models),
            selected_parts_model_path=self._mcmot.parts_model_path,
            parts_model_loaded=bool(self._mcmot.parts_model_loaded),
            parts_confidence_threshold=float(self._mcmot.parts_confidence_threshold),
            parts_nms_enabled=bool(self._mcmot.parts_nms_enabled),
            parts_nms_iou_threshold=float(self._mcmot.parts_nms_iou_threshold),
            selected_arm_model_path=self._mcmot.arm_model_path,
            arm_model_loaded=bool(self._mcmot.arm_model_loaded),
            arm_confidence_threshold=float(self._mcmot.arm_confidence_threshold),
            arm_nms_enabled=bool(self._mcmot.arm_nms_enabled),
            arm_nms_iou_threshold=float(self._mcmot.arm_nms_iou_threshold),
            arm_inference_interval_sec=float(self._mcmot.arm_inference_interval_sec),
            arm_connected=bool(
                self._mcmot.ned2_arm_controller is not None
                and getattr(self._mcmot.ned2_arm_controller, "connected", False)
            ),
            arm_linked_id=linked_arm_id,
            arm_linked_no_control=(
                bool(linked_arm_obj.get("no_control", False))
                if linked_arm_obj is not None
                else None
            ),
            available_models=list(self._mcmot.available_models),
            selected_model_path=self._mcmot.model_path,
            model_loaded=bool(self._mcmot.model_loaded),
            confidence_threshold=float(self._mcmot.confidence_threshold),
            overhead_update_interval_sec=float(self._mcmot.overhead_update_interval_sec),
            calibration_preview_mode=self._mcmot.calibration_preview_mode,
            click_mode=self._click_mode,
            available_click_modes=sorted(self._click_handlers.keys()),
            pending_landmark_pixels=self._serialize_pending_landmarks(),
            pending_arm_base_selections=self._serialize_pending_arm_base_selections(),
        )
        self._frames: dict[str, bytes] = {}
        self._frame_versions: dict[str, int] = {}

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="mcmot-background-service",
        )
        self._thread.start()

    @staticmethod
    def _model_to_dict(model_obj) -> dict:
        if hasattr(model_obj, "model_dump"):
            return model_obj.model_dump()
        return model_obj.dict()

    def _encode_frame(self, frame: np.ndarray, quality: int) -> Optional[bytes]:
        quality = max(20, min(95, int(quality)))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return encoded.tobytes()

    def _serialize_pending_landmarks(self) -> dict[str, list[float]]:
        return {
            str(camera_number): [float(pixel[0]), float(pixel[1])]
            for camera_number, pixel in self._pending_landmark_pixels.items()
        }

    def _serialize_pending_arm_base_selections(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for camera_number, selection in self._pending_arm_base_selections.items():
            payload[str(camera_number)] = {
                "camera_number": int(camera_number),
                "detection_index": int(selection["detection_index"]),
                "keypoint_xy": [
                    float(selection["keypoint_xy"][0]),
                    float(selection["keypoint_xy"][1]),
                ],
                "distance_px": float(selection["distance_px"]),
                "selected_at": float(selection["selected_at"]),
            }
        return payload

    def _store_frame_locked(self, stream_name: str, payload: bytes) -> None:
        self._frames[stream_name] = payload
        self._frame_versions[stream_name] = int(self._frame_versions.get(stream_name, 0)) + 1

    def _remove_frame_locked(self, stream_name: str) -> None:
        self._frames.pop(stream_name, None)
        self._frame_versions.pop(stream_name, None)

    @staticmethod
    def _camera_number_from_stream(stream_name: str) -> Optional[int]:
        if stream_name.startswith("cam") and stream_name[3:].isdigit():
            return int(stream_name[3:])
        return None

    @staticmethod
    def _to_scalar(value):
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def register_click_handler(
        self,
        mode: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        mode_key = str(mode).strip().lower()
        self._click_handlers[mode_key] = handler

    def _resolve_detection_hit(self, camera_number: int, x: float, y: float) -> Optional[dict]:
        camera = self._mcmot.cameras.get(camera_number)
        if camera is None or getattr(camera, "model_plus", None) is None:
            return None

        tracked = getattr(camera.model_plus, "detections_tracked", None)
        xyxy = getattr(tracked, "xyxy", None) if tracked is not None else None
        if xyxy is None:
            return None

        boxes = np.asarray(xyxy)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            return None

        best_idx = None
        best_area = None
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(1.0, (x2 - x1) * (y2 - y1))
                if best_idx is None or area < best_area:
                    best_idx = idx
                    best_area = area

        if best_idx is None:
            return None

        def _value_at(attr_name: str):
            values = getattr(tracked, attr_name, None)
            if values is None:
                return None
            try:
                return values[best_idx]
            except Exception:
                return None

        tracker_id = self._to_scalar(_value_at("tracker_id"))
        class_id = self._to_scalar(_value_at("class_id"))
        confidence = self._to_scalar(_value_at("confidence"))

        hit_box = boxes[best_idx]
        return {
            "camera_number": int(camera_number),
            "detection_index": int(best_idx),
            "track_id": int(tracker_id) if tracker_id is not None else None,
            "class_id": int(class_id) if class_id is not None else None,
            "confidence": float(confidence) if confidence is not None else None,
            "bbox_xyxy": [float(hit_box[0]), float(hit_box[1]), float(hit_box[2]), float(hit_box[3])],
        }

    def _handle_click_none(self, event: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": "record_only",
            "camera_number": event.get("camera_number"),
        }

    def _handle_click_landmark_select(self, event: dict[str, Any]) -> dict[str, Any]:
        camera_number = event.get("camera_number")
        if camera_number is None or camera_number not in self._mcmot.cameras:
            return {
                "action": "ignored",
                "reason": "click must be on a camera stream",
            }
        pixel_xy = [float(event["x"]), float(event["y"])]
        self._pending_landmark_pixels[int(camera_number)] = pixel_xy
        return {
            "action": "set_pending_landmark",
            "camera_number": int(camera_number),
            "pixel_xy": pixel_xy,
        }

    def _resolve_arm_base_hit(
        self,
        camera_number: int,
        x: float,
        y: float,
        max_distance_px: float = 30.0,
    ) -> Optional[dict[str, Any]]:
        detections = self._mcmot.arm_detections_by_camera.get(int(camera_number), [])
        best = None
        best_distance = None
        for detection_idx, detection in enumerate(detections):
            keypoint = detection.get("keypoints", {}).get("base")
            if keypoint is None:
                continue
            kp_xy = keypoint.get("xy")
            if kp_xy is None or len(kp_xy) != 2:
                continue
            bx = float(kp_xy[0])
            by = float(kp_xy[1])
            if not np.isfinite(bx) or not np.isfinite(by):
                continue
            distance = float(np.hypot(float(x) - bx, float(y) - by))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = {
                    "camera_number": int(camera_number),
                    "detection_index": int(detection_idx),
                    "keypoint_xy": [bx, by],
                    "distance_px": float(distance),
                }

        if best is None or best_distance is None or best_distance > float(max_distance_px):
            return None
        return best

    def _handle_click_arm_base_select(self, event: dict[str, Any]) -> dict[str, Any]:
        camera_number = event.get("camera_number")
        if camera_number is None or camera_number not in self._mcmot.cameras:
            return {
                "action": "ignored",
                "reason": "click must be on a camera stream",
            }
        hit = self._resolve_arm_base_hit(
            int(camera_number),
            float(event.get("x", 0.0)),
            float(event.get("y", 0.0)),
        )
        if hit is None:
            return {
                "action": "ignored",
                "reason": "no arm base keypoint near click",
            }

        hit["selected_at"] = float(time.time())
        self._pending_arm_base_selections[int(camera_number)] = hit
        return {
            "action": "set_pending_arm_base",
            **hit,
            "selected_cameras": sorted(self._pending_arm_base_selections.keys()),
        }

    def _serialize_tracks(self, global_tracks: dict) -> list[dict]:
        serialized: list[dict] = []
        for match, values in global_tracks.items():
            cam0_tid = int(match[0]) if match[0] is not None else None
            cam1_tid = int(match[1]) if match[1] is not None else None
            item = {
                "cam0_track_id": cam0_tid,
                "cam1_track_id": cam1_tid,
                "point_xyz": None,
                "color": None,
                "shape": None,
            }
            if values is not None:
                point_3d, color, shape = values
                item["point_xyz"] = np.asarray(point_3d).reshape(-1).astype(float).tolist()
                item["color"] = str(color)
                item["shape"] = str(shape)
            serialized.append(item)
        return serialized

    def _collect_detection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cam_id, camera in self._mcmot.cameras.items():
            num = 0
            model_plus = getattr(camera, "model_plus", None)
            tracked = getattr(model_plus, "detections_tracked", None) if model_plus else None
            if tracked is not None and hasattr(tracked, "tracker_id"):
                try:
                    num = len(tracked.tracker_id.tolist())
                except Exception:
                    num = 0
            counts[f"cam{cam_id}"] = int(num)
        return counts

    def _update_runtime_snapshot(self, loop_ms: float, fps: float):
        self._runtime.last_loop_ms = float(loop_ms)
        self._runtime.fps = float(fps)
        self._runtime.available_streams = sorted(self._frames.keys())
        self._runtime.tracks_3d = self._serialize_tracks(self._mcmot.global_tracks)
        self._runtime.detections_per_camera = self._collect_detection_counts()
        self._runtime.calibration_method = self._mcmot.calibration_method
        self._runtime.mesh_display = bool(self._mcmot.mesh_display)
        self._runtime.available_camera_ids = list(self._mcmot.available_camera_ids)
        self._runtime.selected_camera_ids = list(self._mcmot.selected_camera_ids)
        self._runtime.selected_camera_names = list(self._mcmot.selected_camera_names)
        self._runtime.available_parts_models = list(self._mcmot.available_parts_models)
        self._runtime.available_arm_models = list(self._mcmot.available_arm_models)
        self._runtime.selected_parts_model_path = self._mcmot.parts_model_path
        self._runtime.parts_model_loaded = bool(self._mcmot.parts_model_loaded)
        self._runtime.parts_confidence_threshold = float(self._mcmot.parts_confidence_threshold)
        self._runtime.parts_nms_enabled = bool(self._mcmot.parts_nms_enabled)
        self._runtime.parts_nms_iou_threshold = float(self._mcmot.parts_nms_iou_threshold)
        self._runtime.selected_arm_model_path = self._mcmot.arm_model_path
        self._runtime.arm_model_loaded = bool(self._mcmot.arm_model_loaded)
        self._runtime.arm_confidence_threshold = float(self._mcmot.arm_confidence_threshold)
        self._runtime.arm_nms_enabled = bool(self._mcmot.arm_nms_enabled)
        self._runtime.arm_nms_iou_threshold = float(self._mcmot.arm_nms_iou_threshold)
        self._runtime.arm_inference_interval_sec = float(self._mcmot.arm_inference_interval_sec)
        self._runtime.arm_connected = bool(
            self._mcmot.ned2_arm_controller is not None
            and getattr(self._mcmot.ned2_arm_controller, "connected", False)
        )
        linked_arm_id = (
            sorted(self._mcmot.arm_controller_links.keys())[0]
            if getattr(self._mcmot, "arm_controller_links", None)
            else None
        )
        if linked_arm_id is None and self._runtime.arm_linked_id is not None:
            candidate = int(self._runtime.arm_linked_id)
            if candidate in self._mcmot.arm_objects:
                linked_arm_id = candidate
        self._runtime.arm_linked_id = linked_arm_id
        linked_arm_obj = (
            self._mcmot.arm_objects.get(self._runtime.arm_linked_id)
            if self._runtime.arm_linked_id is not None
            else None
        )
        self._runtime.arm_linked_no_control = (
            bool(linked_arm_obj.get("no_control", False))
            if linked_arm_obj is not None
            else None
        )
        self._runtime.available_models = list(self._mcmot.available_models)
        self._runtime.selected_model_path = self._mcmot.model_path
        self._runtime.model_loaded = bool(self._mcmot.model_loaded)
        self._runtime.confidence_threshold = float(self._mcmot.confidence_threshold)
        self._runtime.overhead_update_interval_sec = float(self._mcmot.overhead_update_interval_sec)
        self._runtime.calibration_preview_mode = self._mcmot.calibration_preview_mode
        self._runtime.click_mode = self._click_mode
        self._runtime.available_click_modes = sorted(self._click_handlers.keys())
        self._runtime.pending_landmark_pixels = self._serialize_pending_landmarks()
        self._runtime.pending_arm_base_selections = self._serialize_pending_arm_base_selections()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._runtime.running)

    def shutdown(self, timeout_sec: float = 5.0) -> tuple[bool, str]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            if alive:
                return False, "Timed out while stopping background loop."
            self._runtime.running = False
            return True, "Background loop stopped."

    def start(self, config: SessionConfig) -> tuple[bool, str]:
        # "start" means attach models and enable annotations.
        with self._lock:
            try:
                if config.camera_device_ids:
                    self._mcmot.set_active_cameras(
                        config.camera_device_ids,
                        config.camera_names,
                    )
                parts_model_path = config.parts_model_path or config.model_path
                parts_conf = (
                    float(config.parts_confidence_threshold)
                    if config.confidence_threshold is None
                    else float(config.confidence_threshold)
                )
                if parts_model_path:
                    self._mcmot.set_parts_model(
                        model_path=parts_model_path,
                        confidence_threshold=parts_conf,
                        tracker_yaml_path=config.tracker_yaml_path,
                    )
                else:
                    self._mcmot.clear_parts_model()
                self._mcmot.set_parts_nms_options(
                    enabled=config.parts_nms_enabled,
                    iou_threshold=config.parts_nms_iou_threshold,
                )

                self._mcmot.set_arm_confidence_threshold(config.arm_confidence_threshold)
                self._mcmot.set_arm_nms_options(
                    enabled=config.arm_nms_enabled,
                    iou_threshold=config.arm_nms_iou_threshold,
                )
                self._mcmot.set_arm_inference_interval(config.arm_inference_interval_sec)
                if config.arm_model_path:
                    self._mcmot.set_arm_model(
                        config.arm_model_path,
                        confidence_threshold=config.arm_confidence_threshold,
                    )
                else:
                    self._mcmot.clear_arm_model()
                self._mcmot.mesh_display = bool(config.mesh_display)
                self._loop_delay_sec = float(config.loop_delay_sec)
                self._jpeg_quality = int(config.jpeg_quality)
                self._display_overhead = bool(config.display_overhead)
                self._mcmot.set_overhead_update_interval(config.overhead_update_interval_sec)
                self._runtime.config = deepcopy(self._model_to_dict(config))
                return True, "Session models updated."
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc)

    def stop(self, timeout_sec: float = 5.0) -> tuple[bool, str]:
        # Keep preview loop running; just detach models.
        with self._lock:
            try:
                self._mcmot.clear_parts_model()
                self._mcmot.clear_arm_model()
                self._runtime.config = None
                return True, "Models detached."
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc)

    def refresh_options(self) -> tuple[bool, str, dict]:
        with self._lock:
            try:
                data = self._mcmot.refresh_options()
                return True, "Options refreshed.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def set_active_cameras(self, camera_device_ids: list[int], camera_names: Optional[list[str]] = None):
        with self._lock:
            try:
                data = self._mcmot.set_active_cameras(camera_device_ids, camera_names)
                self._pending_landmark_pixels = {}
                self._runtime.pending_landmark_pixels = {}
                self._pending_arm_base_selections = {}
                self._runtime.pending_arm_base_selections = {}
                return True, "Active cameras updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def set_parts_model(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        tracker_yaml_path: Optional[str] = None,
    ):
        with self._lock:
            try:
                data = self._mcmot.set_parts_model(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold,
                    tracker_yaml_path=tracker_yaml_path,
                )
                return True, "Parts model loaded.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_parts_confidence_threshold(self, confidence_threshold: float):
        with self._lock:
            try:
                data = self._mcmot.set_parts_confidence_threshold(confidence_threshold)
                self._runtime.confidence_threshold = float(self._mcmot.confidence_threshold)
                self._runtime.parts_confidence_threshold = float(
                    self._mcmot.parts_confidence_threshold
                )
                return True, "Parts confidence threshold updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_parts_nms_options(
        self,
        enabled: bool,
        iou_threshold: float,
    ):
        with self._lock:
            try:
                data = self._mcmot.set_parts_nms_options(
                    enabled=enabled,
                    iou_threshold=iou_threshold,
                )
                self._runtime.parts_nms_enabled = bool(self._mcmot.parts_nms_enabled)
                self._runtime.parts_nms_iou_threshold = float(self._mcmot.parts_nms_iou_threshold)
                return True, "Parts NMS options updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_overhead_interval(self, interval_sec: float):
        with self._lock:
            try:
                data = self._mcmot.set_overhead_update_interval(interval_sec)
                self._runtime.overhead_update_interval_sec = float(
                    self._mcmot.overhead_update_interval_sec
                )
                return True, "Overhead interval updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def clear_parts_model(self):
        with self._lock:
            try:
                data = self._mcmot.clear_parts_model()
                return True, "Parts model cleared.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def set_arm_model(self, model_path: str, confidence_threshold: Optional[float] = None):
        with self._lock:
            try:
                data = self._mcmot.set_arm_model(
                    model_path,
                    confidence_threshold=confidence_threshold,
                )
                return True, "Arm model loaded.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def clear_arm_model(self):
        with self._lock:
            try:
                data = self._mcmot.clear_arm_model()
                self._pending_arm_base_selections = {}
                self._runtime.pending_arm_base_selections = {}
                return True, "Arm model cleared.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_arm_interval(self, interval_sec: float):
        with self._lock:
            try:
                data = self._mcmot.set_arm_inference_interval(interval_sec)
                self._runtime.arm_inference_interval_sec = float(
                    self._mcmot.arm_inference_interval_sec
                )
                return True, "Arm inference interval updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_arm_confidence_threshold(self, confidence_threshold: float):
        with self._lock:
            try:
                data = self._mcmot.set_arm_confidence_threshold(confidence_threshold)
                self._runtime.arm_confidence_threshold = float(
                    self._mcmot.arm_confidence_threshold
                )
                return True, "Arm confidence threshold updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def update_arm_nms_options(
        self,
        enabled: bool,
        iou_threshold: float,
    ):
        with self._lock:
            try:
                data = self._mcmot.set_arm_nms_options(
                    enabled=enabled,
                    iou_threshold=iou_threshold,
                )
                self._runtime.arm_nms_enabled = bool(self._mcmot.arm_nms_enabled)
                self._runtime.arm_nms_iou_threshold = float(self._mcmot.arm_nms_iou_threshold)
                return True, "Arm NMS options updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def connect_arm(
        self,
        ip_address: str,
        port: int = 9090,
        auto_calibrate: bool = True,
        disable_learning_mode: bool = True,
        update_tool: bool = True,
    ):
        with self._lock:
            try:
                data = self._mcmot.connect_ned2_arm(
                    ip_address=ip_address,
                    port=port,
                    auto_calibrate=auto_calibrate,
                    disable_learning_mode=disable_learning_mode,
                    update_tool=update_tool,
                )
                self._runtime.arm_connected = bool(data.get("connected", True))
                return True, "Arm connected.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def _resolve_target_arm_id_locked(
        self,
        arm_id: Optional[int],
        cam0_detection_index: Optional[int] = None,
        cam1_detection_index: Optional[int] = None,
        prefer_existing_within_threshold: bool = True,
        create_new_when_linked_to_added: bool = False,
    ) -> int:
        if arm_id is not None:
            target_arm_id = int(arm_id)
            if target_arm_id not in self._mcmot.arm_objects:
                raise RuntimeError(f"Unknown arm_id {target_arm_id}.")
            return target_arm_id

        if cam0_detection_index is not None and cam1_detection_index is not None:
            selected = self._mcmot.resolve_or_create_arm_from_detection_pair(
                cam0_detection_index=cam0_detection_index,
                cam1_detection_index=cam1_detection_index,
                prefer_existing_within_threshold=prefer_existing_within_threshold,
                create_new_when_linked_to_added=create_new_when_linked_to_added,
            )
            return int(selected["arm_id"])

        active_arms = self._mcmot.get_active_arm_objects(include_missed=False)
        if not active_arms:
            raise RuntimeError(
                "No active matched arm object found. Detect an arm in both cameras first."
            )
        return int(sorted(active_arms.keys())[0])

    def _make_arm_gripper_observer(self, target_arm_id: int, timeout_sec: float):
        last_seen_frame = -1

        def _observe_global_xyz_fn(_robot_xyz, _sample_index):
            nonlocal last_seen_frame
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                with self._lock:
                    arm_obj = self._mcmot.arm_objects.get(target_arm_id)
                    if arm_obj is not None:
                        seen_frame = int(arm_obj.get("last_seen_frame", -1))
                        keypoints_3d = arm_obj.get("keypoints_3d", {})
                        gripper_xyz = keypoints_3d.get("gripper")
                        if gripper_xyz is not None and seen_frame > last_seen_frame:
                            last_seen_frame = seen_frame
                            return [
                                float(gripper_xyz[0]),
                                float(gripper_xyz[1]),
                                float(gripper_xyz[2]),
                            ]
                time.sleep(0.05)
            return None

        return _observe_global_xyz_fn

    def add_arm_to_scene(
        self,
        arm_id: Optional[int] = None,
        cam0_detection_index: Optional[int] = None,
        cam1_detection_index: Optional[int] = None,
        no_control: bool = False,
        run_calibration: bool = True,
        ip_address: Optional[str] = None,
        port: int = 9090,
        x_values: Optional[list[float]] = None,
        y_values: Optional[list[float]] = None,
        z_values: Optional[list[float]] = None,
        orientation_rpy: Optional[list[float]] = None,
        settle_time_sec: float = 0.35,
        allow_scale: bool = True,
        require_all_points: bool = False,
        sample_timeout_sec: float = 2.0,
        auto_calibrate: bool = True,
        disable_learning_mode: bool = True,
        update_tool: bool = True,
    ):
        no_control = bool(no_control)
        run_calibration = bool(run_calibration) and (not no_control)
        try:
            with self._lock:
                target_arm_id = self._resolve_target_arm_id_locked(
                    arm_id=arm_id,
                    cam0_detection_index=cam0_detection_index,
                    cam1_detection_index=cam1_detection_index,
                    prefer_existing_within_threshold=(arm_id is not None),
                    create_new_when_linked_to_added=(arm_id is None),
                )
                self._mcmot.set_arm_object_in_scene(target_arm_id, in_scene=True)
                if cam0_detection_index is not None and cam1_detection_index is not None:
                    self._mcmot.set_arm_user_base_xyz(target_arm_id)
                self._mcmot.set_arm_object_no_control(target_arm_id, no_control=no_control)
                self._mcmot.set_arm_object_calibrated(target_arm_id, calibrated=False)

                if no_control:
                    self._mcmot.unlink_ned2_controller_from_arm(target_arm_id)
                    self._pending_arm_base_selections = {}
                    self._runtime.arm_linked_id = target_arm_id
                    self._runtime.arm_linked_no_control = True
                    self._runtime.arm_connected = bool(
                        self._mcmot.ned2_arm_controller is not None
                        and getattr(self._mcmot.ned2_arm_controller, "connected", False)
                    )
                    self._runtime.pending_arm_base_selections = {}
                    return True, "Arm added to scene with no control.", {
                        "arm_id": target_arm_id,
                        "cam0_detection_index": cam0_detection_index,
                        "cam1_detection_index": cam1_detection_index,
                        "no_control": True,
                        "controller_linked": False,
                        "calibrated": False,
                    }

                if ip_address:
                    self._mcmot.connect_ned2_arm(
                        ip_address=ip_address,
                        port=port,
                        auto_calibrate=auto_calibrate,
                        disable_learning_mode=disable_learning_mode,
                        update_tool=update_tool,
                    )

                controller_connected = bool(
                    self._mcmot.ned2_arm_controller is not None
                    and getattr(self._mcmot.ned2_arm_controller, "connected", False)
                )
                if run_calibration and not controller_connected:
                    raise RuntimeError(
                        "No arm controller connected. Provide an IP address or connect the arm first."
                    )

                if controller_connected:
                    self._mcmot.link_ned2_controller_to_arm(target_arm_id)
                    self._mcmot.sync_arm_origin_from_controller(target_arm_id)
                else:
                    self._mcmot.unlink_ned2_controller_from_arm(target_arm_id)

                self._runtime.arm_connected = controller_connected
                self._runtime.arm_linked_id = target_arm_id
                self._runtime.arm_linked_no_control = False
                timeout_sec = max(0.2, float(sample_timeout_sec))
                self._pending_arm_base_selections = {}
                self._runtime.pending_arm_base_selections = {}

            if not run_calibration:
                message = (
                    "Arm added to scene."
                    if controller_connected
                    else "Arm added to scene without controller connection."
                )
                return True, message, {
                    "arm_id": target_arm_id,
                    "cam0_detection_index": cam0_detection_index,
                    "cam1_detection_index": cam1_detection_index,
                    "no_control": False,
                    "controller_linked": bool(controller_connected),
                    "calibrated": False,
                }

            observe_fn = self._make_arm_gripper_observer(
                target_arm_id=target_arm_id,
                timeout_sec=timeout_sec,
            )
            calibration_data = self._mcmot.calibrate_ned2_arm_grid(
                observe_global_xyz_fn=observe_fn,
                x_values=x_values or [0.15, 0.20, 0.25],
                y_values=y_values or [-0.05, 0.0, 0.05],
                z_values=z_values or [0.08, 0.12, 0.16],
                orientation_rpy=orientation_rpy,
                settle_time_sec=settle_time_sec,
                allow_scale=allow_scale,
                require_all_points=require_all_points,
            )
            with self._lock:
                self._mcmot.set_arm_object_calibrated(target_arm_id, calibrated=True)
                self._mcmot.sync_arm_origin_from_controller(target_arm_id)
                self._runtime.arm_connected = bool(
                    self._mcmot.ned2_arm_controller is not None
                    and getattr(self._mcmot.ned2_arm_controller, "connected", False)
                )
                self._runtime.arm_linked_id = target_arm_id
                self._runtime.arm_linked_no_control = False
            return True, "Arm added to scene and calibrated.", {
                "arm_id": target_arm_id,
                "cam0_detection_index": cam0_detection_index,
                "cam1_detection_index": cam1_detection_index,
                "no_control": False,
                "controller_linked": True,
                "calibrated": True,
                "calibration": calibration_data,
            }
        except Exception as exc:
            with self._lock:
                self._runtime.last_error = str(exc)
            return False, str(exc), {}

    def calibrate_arm_grid(
        self,
        ip_address: Optional[str] = None,
        port: int = 9090,
        arm_id: Optional[int] = None,
        x_values: Optional[list[float]] = None,
        y_values: Optional[list[float]] = None,
        z_values: Optional[list[float]] = None,
        orientation_rpy: Optional[list[float]] = None,
        settle_time_sec: float = 0.35,
        allow_scale: bool = True,
        require_all_points: bool = False,
        sample_timeout_sec: float = 2.0,
        auto_calibrate: bool = True,
        disable_learning_mode: bool = True,
        update_tool: bool = True,
    ):
        return self.add_arm_to_scene(
            arm_id=arm_id,
            no_control=False,
            run_calibration=True,
            ip_address=ip_address,
            port=port,
            x_values=x_values,
            y_values=y_values,
            z_values=z_values,
            orientation_rpy=orientation_rpy,
            settle_time_sec=settle_time_sec,
            allow_scale=allow_scale,
            require_all_points=require_all_points,
            sample_timeout_sec=sample_timeout_sec,
            auto_calibrate=auto_calibrate,
            disable_learning_mode=disable_learning_mode,
            update_tool=update_tool,
        )

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

    def update_confidence_threshold(self, confidence_threshold: float):
        return self.update_parts_confidence_threshold(confidence_threshold)

    def clear_detection_model(self):
        return self.clear_parts_model()

    def set_calibration_preview_mode(self, mode: Optional[str], aruco_positions: str = "square4"):
        with self._lock:
            try:
                data = self._mcmot.set_calibration_preview_mode(mode, aruco_positions=aruco_positions)
                return True, "Calibration preview mode updated.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def calibrate_auto(self, baseline: float = 1.0):
        with self._lock:
            try:
                data = self._mcmot.calibrate_auto_stepwise(baseline=baseline)
                return True, "Auto calibration complete.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def calibrate_aruco(
        self,
        aruco_positions: str = "square4",
        sample_frames: int = 10,
        sample_interval_sec: float = 0.05,
    ):
        with self._lock:
            try:
                data = self._mcmot.calibrate_aruco_stepwise(
                    aruco_positions=aruco_positions,
                    sample_frames=sample_frames,
                    sample_interval_sec=sample_interval_sec,
                )
                return True, "ArUco calibration complete.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def set_click_mode(self, mode: str):
        with self._lock:
            try:
                mode_key = (mode or "none").strip().lower()
                if mode_key not in self._click_handlers:
                    raise ValueError(
                        f"Unknown click mode '{mode_key}'. "
                        f"Available: {sorted(self._click_handlers.keys())}"
                    )
                self._click_mode = mode_key
                self._runtime.click_mode = self._click_mode
                self._runtime.available_click_modes = sorted(self._click_handlers.keys())
                return True, f"Click mode set to '{self._click_mode}'.", {
                    "click_mode": self._click_mode,
                    "available_click_modes": sorted(self._click_handlers.keys()),
                }
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def handle_click(
        self,
        stream_name: str,
        x: float,
        y: float,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        button: str = "left",
        modifiers: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        with self._lock:
            try:
                mode = self._click_mode
                camera_number = self._camera_number_from_stream(stream_name)
                event = {
                    "ts": float(time.time()),
                    "stream_name": str(stream_name),
                    "camera_number": camera_number,
                    "x": float(x),
                    "y": float(y),
                    "image_width": int(image_width) if image_width is not None else None,
                    "image_height": int(image_height) if image_height is not None else None,
                    "button": str(button),
                    "modifiers": list(modifiers or []),
                    "metadata": dict(metadata or {}),
                }

                detection_hit = None
                if camera_number is not None:
                    detection_hit = self._resolve_detection_hit(camera_number, float(x), float(y))

                handler = self._click_handlers.get(mode, self._handle_click_none)
                resolved_target = handler(event)

                last_click = {
                    "mode": mode,
                    "stream_name": event["stream_name"],
                    "camera_number": camera_number,
                    "x": event["x"],
                    "y": event["y"],
                    "detection_hit": detection_hit,
                    "resolved_target": resolved_target,
                    "ts": event["ts"],
                }
                self._runtime.last_click = last_click
                self._runtime.pending_landmark_pixels = self._serialize_pending_landmarks()
                self._runtime.pending_arm_base_selections = self._serialize_pending_arm_base_selections()

                return True, "Click handled.", {
                    "mode": mode,
                    "event": event,
                    "detection_hit": detection_hit,
                    "resolved_target": resolved_target,
                    "pending_landmark_pixels": self._serialize_pending_landmarks(),
                    "pending_arm_base_selections": self._serialize_pending_arm_base_selections(),
                }
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def get_click_state(self):
        with self._lock:
            return True, "Click state.", {
                "click_mode": self._click_mode,
                "available_click_modes": sorted(self._click_handlers.keys()),
                "last_click": deepcopy(self._runtime.last_click),
                "pending_landmark_pixels": self._serialize_pending_landmarks(),
                "pending_arm_base_selections": self._serialize_pending_arm_base_selections(),
            }

    def clear_pending_arm_base_selections(self, camera_number: Optional[int] = None):
        with self._lock:
            try:
                if camera_number is None:
                    self._pending_arm_base_selections = {}
                else:
                    self._pending_arm_base_selections.pop(int(camera_number), None)
                data = {
                    "camera_number": camera_number,
                    "pending_arm_base_selections": self._serialize_pending_arm_base_selections(),
                }
                self._runtime.pending_arm_base_selections = data["pending_arm_base_selections"]
                return True, "Arm base selections cleared.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def add_landmark_observation(self, camera_number: int, pixel_xy, world_xyz):
        with self._lock:
            try:
                resolved_pixel = pixel_xy
                if resolved_pixel is None:
                    resolved_pixel = self._pending_landmark_pixels.get(camera_number)
                    if resolved_pixel is None:
                        raise RuntimeError(
                            f"No pending click pixel for camera {camera_number}. "
                            "Click in the stream or provide pixel_xy explicitly."
                        )

                data = self._mcmot.add_landmark_observation(camera_number, resolved_pixel, world_xyz)
                data["pixel_xy"] = [float(resolved_pixel[0]), float(resolved_pixel[1])]
                self._pending_landmark_pixels.pop(camera_number, None)
                self._runtime.pending_landmark_pixels = self._serialize_pending_landmarks()
                return True, "Landmark point added.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def clear_landmarks(self, camera_number: Optional[int] = None):
        with self._lock:
            try:
                self._mcmot.clear_landmarks(camera_number)
                if camera_number is None:
                    self._pending_landmark_pixels = {}
                else:
                    self._pending_landmark_pixels.pop(camera_number, None)
                self._runtime.pending_landmark_pixels = self._serialize_pending_landmarks()
                return True, "Landmarks cleared.", {"camera_number": camera_number}
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def solve_landmark_calibration(self, camera_number: int):
        with self._lock:
            try:
                data = self._mcmot.solve_landmark_calibration(camera_number)
                return True, "Landmark calibration complete.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def solve_landmark_calibration_all(self, camera_numbers: Optional[list[int]] = None):
        with self._lock:
            try:
                if camera_numbers is None:
                    targets = sorted(self._mcmot.cameras.keys())
                else:
                    targets = [int(c) for c in camera_numbers]
                if not targets:
                    raise RuntimeError("No active cameras available for landmark calibration.")

                results: dict[str, dict] = {}
                failures: list[str] = []
                for camera_number in targets:
                    try:
                        solved = self._mcmot.solve_landmark_calibration(camera_number)
                        results[str(camera_number)] = {"ok": True, **solved}
                    except Exception as exc:
                        err_msg = str(exc)
                        results[str(camera_number)] = {"ok": False, "error": err_msg}
                        failures.append(f"cam{camera_number}: {err_msg}")

                ok = len(failures) == 0
                if not ok:
                    self._runtime.last_error = "; ".join(failures)
                    return False, "Landmark calibration failed for one or more cameras.", {
                        "camera_numbers": targets,
                        "results": results,
                    }

                return True, "Landmark calibration complete for all cameras.", {
                    "camera_numbers": targets,
                    "results": results,
                }
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def get_landmark_state(self, camera_number: int):
        with self._lock:
            try:
                data = self._mcmot.get_landmark_state(camera_number)
                return True, "Landmark state.", data
            except Exception as exc:
                self._runtime.last_error = str(exc)
                return False, str(exc), {}

    def request_recalibration(self) -> tuple[bool, str]:
        # Backward-compatible endpoint. Use explicit calibration endpoints instead.
        with self._lock:
            return False, "Use /api/calibration/* endpoints for step-by-step calibration."

    def get_frame(self, stream_name: str) -> Optional[bytes]:
        with self._lock:
            data = self._frames.get(stream_name)
            return bytes(data) if data is not None else None

    def get_frame_with_version(self, stream_name: str) -> tuple[Optional[bytes], int]:
        with self._lock:
            data = self._frames.get(stream_name)
            version = int(self._frame_versions.get(stream_name, 0))
            return (bytes(data) if data is not None else None), version

    def get_status(self) -> SessionStatus:
        with self._lock:
            started_at = self._runtime.started_at
            uptime = 0.0 if not started_at else max(0.0, time.time() - started_at)
            return SessionStatus(
                running=self._runtime.running,
                started_at=started_at,
                uptime_sec=uptime,
                fps=self._runtime.fps,
                last_loop_ms=self._runtime.last_loop_ms,
                last_error=self._runtime.last_error,
                calibration_method=self._runtime.calibration_method,
                mesh_display=self._runtime.mesh_display,
                available_streams=list(self._runtime.available_streams),
                tracks_3d=deepcopy(self._runtime.tracks_3d),
                detections_per_camera=deepcopy(self._runtime.detections_per_camera),
                available_camera_ids=list(self._runtime.available_camera_ids),
                selected_camera_ids=list(self._runtime.selected_camera_ids),
                selected_camera_names=list(self._runtime.selected_camera_names),
                available_parts_models=list(self._runtime.available_parts_models),
                available_arm_models=list(self._runtime.available_arm_models),
                selected_parts_model_path=self._runtime.selected_parts_model_path,
                parts_model_loaded=bool(self._runtime.parts_model_loaded),
                parts_confidence_threshold=float(self._runtime.parts_confidence_threshold),
                parts_nms_enabled=bool(self._runtime.parts_nms_enabled),
                parts_nms_iou_threshold=float(self._runtime.parts_nms_iou_threshold),
                selected_arm_model_path=self._runtime.selected_arm_model_path,
                arm_model_loaded=bool(self._runtime.arm_model_loaded),
                arm_confidence_threshold=float(self._runtime.arm_confidence_threshold),
                arm_nms_enabled=bool(self._runtime.arm_nms_enabled),
                arm_nms_iou_threshold=float(self._runtime.arm_nms_iou_threshold),
                arm_inference_interval_sec=float(self._runtime.arm_inference_interval_sec),
                arm_connected=bool(self._runtime.arm_connected),
                arm_linked_id=self._runtime.arm_linked_id,
                arm_linked_no_control=self._runtime.arm_linked_no_control,
                available_models=list(self._runtime.available_models),
                selected_model_path=self._runtime.selected_model_path,
                model_loaded=bool(self._runtime.model_loaded),
                confidence_threshold=float(self._runtime.confidence_threshold),
                overhead_update_interval_sec=float(self._runtime.overhead_update_interval_sec),
                calibration_preview_mode=self._runtime.calibration_preview_mode,
                click_mode=self._runtime.click_mode,
                available_click_modes=list(self._runtime.available_click_modes),
                last_click=deepcopy(self._runtime.last_click),
                pending_landmark_pixels=deepcopy(self._runtime.pending_landmark_pixels),
                pending_arm_base_selections=deepcopy(self._runtime.pending_arm_base_selections),
                config=deepcopy(self._runtime.config),
            )

    def _run_loop(self) -> None:
        fps_ema = 0.0
        while not self._stop_event.is_set():
            loop_start = time.time()
            try:
                with self._lock:
                    self._mcmot.capture_cameras_frames()
                    self._mcmot.update_cameras_tracks()
                    self._mcmot.match_global_tracks()
                    if self._mcmot.mesh_display:
                        self._mcmot.update_scene_mesh_display()

                    for cam_id in sorted(self._mcmot.cameras.keys()):
                        stream_name = f"cam{cam_id}"
                        frame = self._mcmot.annotated_frame_from_camera(cam_id)
                        if frame is not None:
                            jpg = self._encode_frame(frame, self._jpeg_quality)
                            if jpg is not None:
                                self._store_frame_locked(stream_name, jpg)
                                continue
                        self._remove_frame_locked(stream_name)

                    if self._display_overhead:
                        now = time.time()
                        interval = float(self._mcmot.overhead_update_interval_sec)
                        if (
                            self._last_overhead_jpg is None
                            or (now - self._last_overhead_ts) >= interval
                        ):
                            overhead = self._mcmot.plot_overhead()
                            if overhead is not None:
                                jpg = self._encode_frame(overhead, self._jpeg_quality)
                                if jpg is not None:
                                    self._last_overhead_jpg = jpg
                                    self._last_overhead_ts = now
                                    self._store_frame_locked("overhead", jpg)
                    else:
                        self._last_overhead_jpg = None
                        self._remove_frame_locked("overhead")

                    loop_ms = (time.time() - loop_start) * 1000.0
                    fps = 1000.0 / max(loop_ms, 1e-3)
                    fps_ema = fps if fps_ema <= 0.0 else (0.9 * fps_ema + 0.1 * fps)
                    self._update_runtime_snapshot(loop_ms, fps_ema)
                    self._runtime.last_error = None
            except Exception as exc:
                with self._lock:
                    self._runtime.last_error = str(exc)
            finally:
                if self._loop_delay_sec > 0:
                    time.sleep(self._loop_delay_sec)
                else:
                    time.sleep(0.001)

        with self._lock:
            self._mcmot.close()
            self._runtime.running = False
