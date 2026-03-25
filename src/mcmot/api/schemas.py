from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _project_root() -> Path:
    # .../src/mcmot/api/schemas.py -> project root
    return Path(__file__).resolve().parents[3]


def default_model_path() -> str:
    return str(_project_root() / "src" / "mcmot" / "models" / "parts" / "parts_251016_small.pt")


def default_tracker_yaml() -> str:
    return str(_project_root() / "src" / "mcmot" / "config" / "bytetrack.yaml")


class SessionConfig(BaseModel):
    parts_model_path: Optional[str] = None
    model_path: Optional[str] = None  # Backward-compatible alias.
    camera_device_ids: list[int] = Field(default_factory=lambda: [4, 6])
    camera_names: Optional[list[str]] = None
    parts_confidence_threshold: float = 0.25
    confidence_threshold: Optional[float] = None  # Backward-compatible alias.
    parts_nms_enabled: bool = True
    parts_nms_iou_threshold: float = 0.7
    arm_model_path: Optional[str] = None
    arm_confidence_threshold: float = 0.25
    arm_nms_enabled: bool = True
    arm_nms_iou_threshold: float = 0.4
    arm_inference_interval_sec: float = 0.5
    tracker_yaml_path: str = Field(default_factory=default_tracker_yaml)
    calibration_method: Optional[Literal["auto", "aruco", "landmarks"]] = "auto"
    aruco_positions: Optional[str] = None
    mesh_display: bool = False
    mesh_export_path: str = "./artifacts/scene_mesh.glb"
    display_overhead: bool = True
    overhead_update_interval_sec: float = 2.0
    loop_delay_sec: float = 0.0
    jpeg_quality: int = 80


class CameraSelectionRequest(BaseModel):
    camera_device_ids: list[int]
    camera_names: Optional[list[str]] = None


class ModelSelectionRequest(BaseModel):
    model_path: str = Field(default_factory=default_model_path)
    confidence_threshold: Optional[float] = None
    tracker_yaml_path: Optional[str] = None


class PartsModelSelectionRequest(BaseModel):
    model_path: str
    confidence_threshold: Optional[float] = None
    tracker_yaml_path: Optional[str] = None


class ArmModelSelectionRequest(BaseModel):
    model_path: str
    confidence_threshold: Optional[float] = None


class ConfidenceUpdateRequest(BaseModel):
    confidence_threshold: float


class PartsNmsRequest(BaseModel):
    enabled: bool = True
    iou_threshold: float = 0.7


class ArmNmsRequest(BaseModel):
    enabled: bool = True
    iou_threshold: float = 0.4


class OverheadIntervalRequest(BaseModel):
    interval_sec: float


class ArmIntervalRequest(BaseModel):
    interval_sec: float


class ArmConnectRequest(BaseModel):
    ip_address: str
    port: int = 9090
    auto_calibrate: bool = True
    disable_learning_mode: bool = True
    update_tool: bool = True


class ArmCalibrationGridRequest(BaseModel):
    ip_address: Optional[str] = None
    port: int = 9090
    arm_id: Optional[int] = None
    x_values: list[float] = Field(default_factory=lambda: [0.15, 0.20, 0.25])
    y_values: list[float] = Field(default_factory=lambda: [-0.05, 0.0, 0.05])
    z_values: list[float] = Field(default_factory=lambda: [0.08, 0.12, 0.16])
    orientation_rpy: Optional[list[float]] = None
    settle_time_sec: float = 0.35
    allow_scale: bool = True
    require_all_points: bool = False
    sample_timeout_sec: float = 2.0
    auto_calibrate: bool = True
    disable_learning_mode: bool = True
    update_tool: bool = True


class ArmAddRequest(BaseModel):
    ip_address: Optional[str] = None
    port: int = 9090
    arm_id: Optional[int] = None
    cam0_detection_index: Optional[int] = None
    cam1_detection_index: Optional[int] = None
    no_control: bool = False
    run_calibration: bool = True
    x_values: list[float] = Field(default_factory=lambda: [0.15, 0.20, 0.25])
    y_values: list[float] = Field(default_factory=lambda: [-0.05, 0.0, 0.05])
    z_values: list[float] = Field(default_factory=lambda: [0.08, 0.12, 0.16])
    orientation_rpy: Optional[list[float]] = None
    settle_time_sec: float = 0.35
    allow_scale: bool = True
    require_all_points: bool = False
    sample_timeout_sec: float = 2.0
    auto_calibrate: bool = True
    disable_learning_mode: bool = True
    update_tool: bool = True


class CalibrationPreviewRequest(BaseModel):
    mode: Optional[Literal["aruco", "landmarks"]] = None
    aruco_positions: str = "square4"


class AutoCalibrationRequest(BaseModel):
    baseline: float = 1.0


class ArucoCalibrationRequest(BaseModel):
    aruco_positions: str = "square4"
    sample_frames: int = 10
    sample_interval_sec: float = 0.05


class LandmarkAddRequest(BaseModel):
    camera_number: int
    pixel_xy: Optional[list[float]] = None
    world_xyz: list[float]


class LandmarkSolveRequest(BaseModel):
    camera_number: int


class LandmarkSolveAllRequest(BaseModel):
    camera_numbers: Optional[list[int]] = None


class ClickModeRequest(BaseModel):
    mode: str = "none"


class ClickEventRequest(BaseModel):
    stream_name: str
    x: float
    y: float
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    button: Literal["left", "middle", "right"] = "left"
    modifiers: list[Literal["shift", "ctrl", "alt", "meta"]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStatus(BaseModel):
    running: bool
    started_at: Optional[float] = None
    uptime_sec: float = 0.0
    fps: float = 0.0
    last_loop_ms: float = 0.0
    last_error: Optional[str] = None
    calibration_method: Optional[str] = None
    mesh_display: bool = False
    available_streams: list[str] = Field(default_factory=list)
    tracks_3d: list[dict] = Field(default_factory=list)
    detections_per_camera: dict[str, int] = Field(default_factory=dict)
    available_camera_ids: list[int] = Field(default_factory=list)
    selected_camera_ids: list[int] = Field(default_factory=list)
    selected_camera_names: list[str] = Field(default_factory=list)
    available_parts_models: list[str] = Field(default_factory=list)
    available_arm_models: list[str] = Field(default_factory=list)
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
    available_models: list[str] = Field(default_factory=list)
    selected_model_path: Optional[str] = None
    model_loaded: bool = False
    confidence_threshold: float = 0.25
    overhead_update_interval_sec: float = 2.0
    calibration_preview_mode: Optional[str] = None
    click_mode: str = "none"
    available_click_modes: list[str] = Field(default_factory=list)
    last_click: Optional[dict] = None
    pending_landmark_pixels: dict[str, list[float]] = Field(default_factory=dict)
    pending_arm_base_selections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    config: Optional[dict] = None


class StartSessionResponse(BaseModel):
    started: bool
    message: str
    status: SessionStatus


class StopSessionResponse(BaseModel):
    stopped: bool
    message: str
    status: SessionStatus


class RecalibrateResponse(BaseModel):
    accepted: bool
    message: str
    status: SessionStatus


class ActionResponse(BaseModel):
    ok: bool
    message: str
    status: SessionStatus
    data: Optional[dict] = None
