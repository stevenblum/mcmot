from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .schemas import (
    ActionResponse,
    ArmAddRequest,
    ArmCalibrationGridRequest,
    ArmConnectRequest,
    ArmIntervalRequest,
    ArmNmsRequest,
    ArmModelSelectionRequest,
    ArucoCalibrationRequest,
    AutoCalibrationRequest,
    CalibrationPreviewRequest,
    CameraSelectionRequest,
    ConfidenceUpdateRequest,
    OverheadIntervalRequest,
    ClickEventRequest,
    ClickModeRequest,
    LandmarkAddRequest,
    LandmarkSolveAllRequest,
    LandmarkSolveRequest,
    ModelSelectionRequest,
    PartsNmsRequest,
    PartsModelSelectionRequest,
    RecalibrateResponse,
    SessionConfig,
    SessionStatus,
    StartSessionResponse,
    StopSessionResponse,
)
from .service import MCMOTBackgroundService


app = FastAPI(title="MCMOT API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = MCMOTBackgroundService()


def _model_to_dict(model_obj) -> dict:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()


@app.on_event("shutdown")
def on_shutdown() -> None:
    service.shutdown(timeout_sec=3.0)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "running": service.is_running()}


@app.get("/api/session/status", response_model=SessionStatus)
def session_status() -> SessionStatus:
    return service.get_status()


@app.post("/api/session/start", response_model=StartSessionResponse)
def session_start(config: SessionConfig) -> StartSessionResponse:
    ok, message = service.start(config)
    status = service.get_status()
    return StartSessionResponse(started=ok, message=message, status=status)


@app.post("/api/session/stop", response_model=StopSessionResponse)
def session_stop() -> StopSessionResponse:
    ok, message = service.stop()
    status = service.get_status()
    return StopSessionResponse(stopped=ok, message=message, status=status)


@app.post("/api/session/recalibrate", response_model=RecalibrateResponse)
def session_recalibrate() -> RecalibrateResponse:
    ok, message = service.request_recalibration()
    status = service.get_status()
    return RecalibrateResponse(accepted=ok, message=message, status=status)


def _action(ok: bool, message: str, data: dict | None = None) -> ActionResponse:
    return ActionResponse(ok=ok, message=message, data=data, status=service.get_status())


@app.post("/api/options/refresh", response_model=ActionResponse)
def options_refresh() -> ActionResponse:
    ok, message, data = service.refresh_options()
    return _action(ok, message, data)


@app.post("/api/cameras/select", response_model=ActionResponse)
def cameras_select(req: CameraSelectionRequest) -> ActionResponse:
    ok, message, data = service.set_active_cameras(req.camera_device_ids, req.camera_names)
    return _action(ok, message, data)


@app.post("/api/model/parts/select", response_model=ActionResponse)
def parts_model_select(req: PartsModelSelectionRequest) -> ActionResponse:
    ok, message, data = service.set_parts_model(
        req.model_path,
        req.confidence_threshold,
        req.tracker_yaml_path,
    )
    return _action(ok, message, data)


@app.post("/api/model/parts/confidence", response_model=ActionResponse)
def parts_model_confidence(req: ConfidenceUpdateRequest) -> ActionResponse:
    ok, message, data = service.update_parts_confidence_threshold(req.confidence_threshold)
    return _action(ok, message, data)


@app.post("/api/model/parts/nms", response_model=ActionResponse)
def parts_model_nms(req: PartsNmsRequest) -> ActionResponse:
    ok, message, data = service.update_parts_nms_options(
        enabled=req.enabled,
        iou_threshold=req.iou_threshold,
    )
    return _action(ok, message, data)


@app.post("/api/model/parts/clear", response_model=ActionResponse)
def parts_model_clear() -> ActionResponse:
    ok, message, data = service.clear_parts_model()
    return _action(ok, message, data)


@app.post("/api/model/arm/select", response_model=ActionResponse)
def arm_model_select(req: ArmModelSelectionRequest) -> ActionResponse:
    ok, message, data = service.set_arm_model(
        req.model_path,
        req.confidence_threshold,
    )
    return _action(ok, message, data)


@app.post("/api/model/arm/confidence", response_model=ActionResponse)
def arm_model_confidence(req: ConfidenceUpdateRequest) -> ActionResponse:
    ok, message, data = service.update_arm_confidence_threshold(req.confidence_threshold)
    return _action(ok, message, data)


@app.post("/api/model/arm/nms", response_model=ActionResponse)
def arm_model_nms(req: ArmNmsRequest) -> ActionResponse:
    ok, message, data = service.update_arm_nms_options(
        enabled=req.enabled,
        iou_threshold=req.iou_threshold,
    )
    return _action(ok, message, data)


@app.post("/api/model/arm/interval", response_model=ActionResponse)
def arm_model_interval(req: ArmIntervalRequest) -> ActionResponse:
    ok, message, data = service.update_arm_interval(req.interval_sec)
    return _action(ok, message, data)


@app.post("/api/model/arm/clear", response_model=ActionResponse)
def arm_model_clear() -> ActionResponse:
    ok, message, data = service.clear_arm_model()
    return _action(ok, message, data)


@app.post("/api/arm/connect", response_model=ActionResponse)
def arm_connect(req: ArmConnectRequest) -> ActionResponse:
    ok, message, data = service.connect_arm(
        ip_address=req.ip_address,
        port=req.port,
        auto_calibrate=req.auto_calibrate,
        disable_learning_mode=req.disable_learning_mode,
        update_tool=req.update_tool,
    )
    return _action(ok, message, data)


@app.post("/api/arm/add", response_model=ActionResponse)
def arm_add(req: ArmAddRequest) -> ActionResponse:
    ok, message, data = service.add_arm_to_scene(
        arm_id=req.arm_id,
        cam0_detection_index=req.cam0_detection_index,
        cam1_detection_index=req.cam1_detection_index,
        no_control=req.no_control,
        run_calibration=req.run_calibration,
        ip_address=req.ip_address,
        port=req.port,
        x_values=req.x_values,
        y_values=req.y_values,
        z_values=req.z_values,
        orientation_rpy=req.orientation_rpy,
        settle_time_sec=req.settle_time_sec,
        allow_scale=req.allow_scale,
        require_all_points=req.require_all_points,
        sample_timeout_sec=req.sample_timeout_sec,
        auto_calibrate=req.auto_calibrate,
        disable_learning_mode=req.disable_learning_mode,
        update_tool=req.update_tool,
    )
    return _action(ok, message, data)


@app.post("/api/arm/base-selection/clear", response_model=ActionResponse)
def arm_base_selection_clear(camera_number: int | None = None) -> ActionResponse:
    ok, message, data = service.clear_pending_arm_base_selections(camera_number)
    return _action(ok, message, data)


@app.post("/api/arm/calibrate-grid", response_model=ActionResponse)
def arm_calibrate_grid(req: ArmCalibrationGridRequest) -> ActionResponse:
    ok, message, data = service.calibrate_arm_grid(
        ip_address=req.ip_address,
        port=req.port,
        arm_id=req.arm_id,
        x_values=req.x_values,
        y_values=req.y_values,
        z_values=req.z_values,
        orientation_rpy=req.orientation_rpy,
        settle_time_sec=req.settle_time_sec,
        allow_scale=req.allow_scale,
        require_all_points=req.require_all_points,
        sample_timeout_sec=req.sample_timeout_sec,
        auto_calibrate=req.auto_calibrate,
        disable_learning_mode=req.disable_learning_mode,
        update_tool=req.update_tool,
    )
    return _action(ok, message, data)


@app.post("/api/overhead/interval", response_model=ActionResponse)
def overhead_interval(req: OverheadIntervalRequest) -> ActionResponse:
    ok, message, data = service.update_overhead_interval(req.interval_sec)
    return _action(ok, message, data)


# Backward-compatible model endpoints map to parts model controls.
@app.post("/api/model/select", response_model=ActionResponse)
def model_select(req: ModelSelectionRequest) -> ActionResponse:
    ok, message, data = service.set_detection_model(
        req.model_path,
        req.confidence_threshold,
        req.tracker_yaml_path,
    )
    return _action(ok, message, data)


@app.post("/api/model/confidence", response_model=ActionResponse)
def model_confidence(req: ConfidenceUpdateRequest) -> ActionResponse:
    ok, message, data = service.update_confidence_threshold(req.confidence_threshold)
    return _action(ok, message, data)


@app.post("/api/model/clear", response_model=ActionResponse)
def model_clear() -> ActionResponse:
    ok, message, data = service.clear_detection_model()
    return _action(ok, message, data)


@app.post("/api/calibration/preview", response_model=ActionResponse)
def calibration_preview(req: CalibrationPreviewRequest) -> ActionResponse:
    ok, message, data = service.set_calibration_preview_mode(req.mode, aruco_positions=req.aruco_positions)
    return _action(ok, message, data)


@app.post("/api/calibration/auto", response_model=ActionResponse)
def calibration_auto(req: AutoCalibrationRequest) -> ActionResponse:
    ok, message, data = service.calibrate_auto(baseline=req.baseline)
    return _action(ok, message, data)


@app.post("/api/calibration/aruco", response_model=ActionResponse)
def calibration_aruco(req: ArucoCalibrationRequest) -> ActionResponse:
    ok, message, data = service.calibrate_aruco(
        aruco_positions=req.aruco_positions,
        sample_frames=req.sample_frames,
        sample_interval_sec=req.sample_interval_sec,
    )
    return _action(ok, message, data)


@app.post("/api/calibration/landmarks/add", response_model=ActionResponse)
def calibration_landmark_add(req: LandmarkAddRequest) -> ActionResponse:
    ok, message, data = service.add_landmark_observation(
        camera_number=req.camera_number,
        pixel_xy=req.pixel_xy,
        world_xyz=req.world_xyz,
    )
    return _action(ok, message, data)


@app.post("/api/calibration/landmarks/clear", response_model=ActionResponse)
def calibration_landmark_clear(camera_number: int | None = None) -> ActionResponse:
    ok, message, data = service.clear_landmarks(camera_number)
    return _action(ok, message, data)


@app.post("/api/calibration/landmarks/solve", response_model=ActionResponse)
def calibration_landmark_solve(req: LandmarkSolveRequest) -> ActionResponse:
    ok, message, data = service.solve_landmark_calibration(req.camera_number)
    return _action(ok, message, data)


@app.post("/api/calibration/landmarks/solve-all", response_model=ActionResponse)
def calibration_landmark_solve_all(req: LandmarkSolveAllRequest) -> ActionResponse:
    ok, message, data = service.solve_landmark_calibration_all(req.camera_numbers)
    return _action(ok, message, data)


@app.get("/api/calibration/landmarks/{camera_number}", response_model=ActionResponse)
def calibration_landmark_state(camera_number: int) -> ActionResponse:
    ok, message, data = service.get_landmark_state(camera_number)
    return _action(ok, message, data)


@app.post("/api/ui/click-mode", response_model=ActionResponse)
def ui_click_mode(req: ClickModeRequest) -> ActionResponse:
    ok, message, data = service.set_click_mode(req.mode)
    return _action(ok, message, data)


@app.post("/api/ui/click", response_model=ActionResponse)
def ui_click(req: ClickEventRequest) -> ActionResponse:
    ok, message, data = service.handle_click(
        stream_name=req.stream_name,
        x=req.x,
        y=req.y,
        image_width=req.image_width,
        image_height=req.image_height,
        button=req.button,
        modifiers=req.modifiers,
        metadata=req.metadata,
    )
    return _action(ok, message, data)


@app.get("/api/ui/click-state", response_model=ActionResponse)
def ui_click_state() -> ActionResponse:
    ok, message, data = service.get_click_state()
    return _action(ok, message, data)


@app.get("/api/frame/{stream_name}.jpg")
def frame_snapshot(stream_name: str) -> Response:
    frame = service.get_frame(stream_name)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"No frame available for stream '{stream_name}'.")
    return Response(content=frame, media_type="image/jpeg")


def _mjpeg_generator(stream_name: str) -> AsyncGenerator[bytes, None]:
    async def generator() -> AsyncGenerator[bytes, None]:
        boundary = b"--frame"
        last_version = -1
        while True:
            frame, version = service.get_frame_with_version(stream_name)
            if frame is not None and version != last_version:
                headers = (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode("ascii")
                    + b"\r\n\r\n"
                )
                yield headers + frame + b"\r\n"
                last_version = version
            await asyncio.sleep(0.05)

    return generator()


@app.get("/api/stream/{stream_name}.mjpeg")
def frame_stream(stream_name: str) -> StreamingResponse:
    media_type = "multipart/x-mixed-replace; boundary=frame"
    return StreamingResponse(_mjpeg_generator(stream_name), media_type=media_type)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            status = _model_to_dict(service.get_status())
            status["ts"] = time.time()
            await websocket.send_json(status)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"service": "mcmot-api", "docs": "/docs", "health": "/api/health"})
