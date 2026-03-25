from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np


class Ned2ArmController:
    """
    NIRYO Ned2 controller with robot<->global calibration support.

    This class wraps `pyniryo2` for connection and motion, and adds a calibration
    layer that learns a similarity transform between the Ned2 robot frame and the
    tracker global frame:
        global_xyz ~= scale * (R @ robot_xyz) + t

    Notes:
    - Robot arm pose units follow PyNiryo2 conventions:
      xyz in meters, roll/pitch/yaw in radians.
    - The calibration callback must return global XYZ in your tracker units.
      If those units are not meters, `allow_scale=True` should be used.
    """

    def __init__(
        self,
        default_orientation_rpy: Iterable[float] = (0.0, 1.57, 0.0),
        allow_scale_by_default: bool = True,
    ):
        self.default_orientation_rpy = tuple(float(v) for v in default_orientation_rpy)
        if len(self.default_orientation_rpy) != 3:
            raise ValueError("default_orientation_rpy must have 3 values: roll, pitch, yaw.")

        self.allow_scale_by_default = bool(allow_scale_by_default)

        self.ip_address: Optional[str] = None
        self.port: int = 9090
        self.robot = None
        self.connected: bool = False

        self._rotation_robot_to_global = np.eye(3, dtype=np.float64)
        self._translation_robot_to_global = np.zeros(3, dtype=np.float64)
        self._scale_robot_to_global = 1.0
        self.transform_ready = False
        self.transform_source: Optional[str] = None
        self.transform_rmse: Optional[float] = None
        self.last_calibration_summary: Optional[dict[str, Any]] = None

    @staticmethod
    def _load_niryo_robot_class():
        try:
            from pyniryo2 import NiryoRobot  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pyniryo2 is required for Ned2ArmController. Install it with: pip install pyniryo2"
            ) from exc
        return NiryoRobot

    @staticmethod
    def _as_xyz(point_xyz: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(point_xyz), dtype=np.float64).reshape(-1)
        if arr.size != 3 or not np.isfinite(arr).all():
            raise ValueError("Expected a finite XYZ point with exactly 3 values.")
        return arr

    @staticmethod
    def _as_pose6(pose: Iterable[float]) -> list[float]:
        arr = np.asarray(list(pose), dtype=np.float64).reshape(-1)
        if arr.size != 6 or not np.isfinite(arr).all():
            raise ValueError("Expected a finite 6-DoF pose: [x,y,z,roll,pitch,yaw].")
        return arr.astype(float).tolist()

    @staticmethod
    def _pose_object_to_list(pose_obj) -> list[float]:
        if pose_obj is None:
            raise RuntimeError("Robot returned an empty pose.")
        if hasattr(pose_obj, "to_list"):
            pose = pose_obj.to_list()
            return Ned2ArmController._as_pose6(pose)
        return Ned2ArmController._as_pose6(pose_obj)

    def connect(
        self,
        ip_address: str,
        port: int = 9090,
        auto_calibrate: bool = True,
        disable_learning_mode: bool = True,
        update_tool: bool = True,
    ) -> dict[str, Any]:
        NiryoRobot = self._load_niryo_robot_class()
        ip_address = str(ip_address).strip()
        if not ip_address:
            raise ValueError("ip_address is required.")

        if self.connected:
            self.disconnect()

        self.robot = NiryoRobot(ip_address=ip_address, port=int(port))
        self.ip_address = ip_address
        self.port = int(port)
        self.connected = True

        if auto_calibrate:
            self.calibrate_auto()
        if disable_learning_mode:
            self.set_learning_mode(False)
        if update_tool:
            self.update_tool()

        return {
            "connected": self.connected,
            "ip_address": self.ip_address,
            "port": self.port,
        }

    def disconnect(self) -> dict[str, Any]:
        if self.robot is not None:
            try:
                self.robot.end()
            except Exception:
                pass
        self.robot = None
        self.connected = False
        self.ip_address = None
        return {"connected": self.connected}

    def ensure_connected(self):
        if not self.connected or self.robot is None:
            raise RuntimeError("Ned2 arm is not connected. Call connect(ip_address) first.")

    def calibrate_auto(self) -> None:
        self.ensure_connected()
        self.robot.arm.calibrate_auto()

    def request_new_calibration(self) -> None:
        self.ensure_connected()
        self.robot.arm.request_new_calibration()

    def set_learning_mode(self, enabled: bool) -> None:
        self.ensure_connected()
        self.robot.arm.set_learning_mode(bool(enabled))

    def move_joints(self, joints: Iterable[float]) -> list[float]:
        self.ensure_connected()
        joints_list = self._as_pose6(joints)
        self.robot.arm.move_joints(joints_list)
        return joints_list

    def move_robot_pose(
        self,
        pose: Iterable[float],
        frame: str = "",
    ) -> list[float]:
        self.ensure_connected()
        pose_list = self._as_pose6(pose)
        self.robot.arm.move_pose(pose_list, frame=frame)
        return pose_list

    def move_robot_xyz(
        self,
        x: float,
        y: float,
        z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        frame: str = "",
    ) -> list[float]:
        r = self.default_orientation_rpy[0] if roll is None else float(roll)
        p = self.default_orientation_rpy[1] if pitch is None else float(pitch)
        yw = self.default_orientation_rpy[2] if yaw is None else float(yaw)
        pose = [float(x), float(y), float(z), r, p, yw]
        return self.move_robot_pose(pose, frame=frame)

    def get_robot_pose(self) -> list[float]:
        self.ensure_connected()
        pose = self.robot.arm.get_pose()
        return self._pose_object_to_list(pose)

    def update_tool(self) -> None:
        self.ensure_connected()
        self.robot.tool.update_tool()

    def grasp(self) -> None:
        self.ensure_connected()
        self.robot.tool.grasp_with_tool()

    def release(self) -> None:
        self.ensure_connected()
        self.robot.tool.release_with_tool()

    def open_gripper(
        self,
        speed: int = 500,
        max_torque_percentage: int = 100,
        hold_torque_percentage: int = 30,
    ) -> None:
        self.ensure_connected()
        self.robot.tool.open_gripper(
            speed=int(speed),
            max_torque_percentage=int(max_torque_percentage),
            hold_torque_percentage=int(hold_torque_percentage),
        )

    def close_gripper(
        self,
        speed: int = 500,
        max_torque_percentage: int = 100,
        hold_torque_percentage: int = 30,
    ) -> None:
        self.ensure_connected()
        self.robot.tool.close_gripper(
            speed=int(speed),
            max_torque_percentage=int(max_torque_percentage),
            hold_torque_percentage=int(hold_torque_percentage),
        )

    def clear_transform(self) -> None:
        self._rotation_robot_to_global = np.eye(3, dtype=np.float64)
        self._translation_robot_to_global = np.zeros(3, dtype=np.float64)
        self._scale_robot_to_global = 1.0
        self.transform_ready = False
        self.transform_source = None
        self.transform_rmse = None
        self.last_calibration_summary = None

    def set_transform(
        self,
        rotation_robot_to_global: Iterable[Iterable[float]],
        translation_robot_to_global: Iterable[float],
        scale_robot_to_global: float = 1.0,
        source: str = "manual",
        rmse: Optional[float] = None,
    ) -> dict[str, Any]:
        R = np.asarray(rotation_robot_to_global, dtype=np.float64).reshape(3, 3)
        t = self._as_xyz(translation_robot_to_global)
        s = float(scale_robot_to_global)
        if not np.isfinite(R).all() or not np.isfinite(s) or s == 0.0:
            raise ValueError("Invalid transform values.")

        self._rotation_robot_to_global = R
        self._translation_robot_to_global = t
        self._scale_robot_to_global = s
        self.transform_ready = True
        self.transform_source = str(source)
        self.transform_rmse = None if rmse is None else float(rmse)

        return self.get_transform_dict()

    def get_transform_matrix(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._scale_robot_to_global * self._rotation_robot_to_global
        T[:3, 3] = self._translation_robot_to_global
        return T

    def get_transform_dict(self) -> dict[str, Any]:
        return {
            "transform_ready": bool(self.transform_ready),
            "source": self.transform_source,
            "rmse": self.transform_rmse,
            "scale_robot_to_global": float(self._scale_robot_to_global),
            "rotation_robot_to_global": self._rotation_robot_to_global.tolist(),
            "translation_robot_to_global": self._translation_robot_to_global.tolist(),
            "matrix_robot_to_global": self.get_transform_matrix().tolist(),
        }

    def save_transform_json(self, path: str | Path) -> dict[str, Any]:
        payload = self.get_transform_dict()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"path": str(out_path), **payload}

    def load_transform_json(self, path: str | Path) -> dict[str, Any]:
        in_path = Path(path)
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        self.set_transform(
            payload["rotation_robot_to_global"],
            payload["translation_robot_to_global"],
            payload.get("scale_robot_to_global", 1.0),
            source=payload.get("source", "file"),
            rmse=payload.get("rmse"),
        )
        return {"path": str(in_path), **self.get_transform_dict()}

    def robot_to_global_xyz(self, robot_xyz: Iterable[float]) -> np.ndarray:
        if not self.transform_ready:
            raise RuntimeError("Robot-to-global transform is not calibrated yet.")
        p = self._as_xyz(robot_xyz)
        return self._scale_robot_to_global * (self._rotation_robot_to_global @ p) + self._translation_robot_to_global

    def global_to_robot_xyz(self, global_xyz: Iterable[float]) -> np.ndarray:
        if not self.transform_ready:
            raise RuntimeError("Robot-to-global transform is not calibrated yet.")
        g = self._as_xyz(global_xyz)
        return self._rotation_robot_to_global.T @ ((g - self._translation_robot_to_global) / self._scale_robot_to_global)

    def move_global_xyz(
        self,
        x: float,
        y: float,
        z: float,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        frame: str = "",
    ) -> dict[str, Any]:
        robot_xyz = self.global_to_robot_xyz([x, y, z])
        pose = self.move_robot_xyz(
            float(robot_xyz[0]),
            float(robot_xyz[1]),
            float(robot_xyz[2]),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            frame=frame,
        )
        return {
            "global_xyz": [float(x), float(y), float(z)],
            "robot_xyz": [float(robot_xyz[0]), float(robot_xyz[1]), float(robot_xyz[2])],
            "robot_pose": pose,
        }

    def _global_units_per_inch(self) -> float:
        """
        Convert inch offsets into global-frame units using the calibrated scale.

        scale_robot_to_global is [global_units / meter], so:
          1 inch = 0.0254 m = scale * 0.0254 global units
        """
        if not self.transform_ready:
            raise RuntimeError("Robot-to-global transform is not calibrated yet.")
        global_units_per_inch = abs(float(self._scale_robot_to_global)) * 0.0254
        if not np.isfinite(global_units_per_inch) or global_units_per_inch <= 1e-9:
            raise RuntimeError("Invalid transform scale; cannot convert inches to global units.")
        return global_units_per_inch

    def object_pick_global(
        self,
        x: float,
        y: float,
        z: float,
        observe_gripper_global_xyz_fn: Optional[Callable[[], Optional[Iterable[float]]]] = None,
        force_vision_update_fn: Optional[Callable[[], Any]] = None,
        orientation_rpy: Optional[Iterable[float]] = None,
        frame: str = "",
        approach_height_in: float = 2.0,
        descent_checkpoints_in: Iterable[float] = (1.0, 0.5, 0.2, 0.0),
        verify_tolerance_in: float = 1.0,
        verify_retries: int = 4,
        verify_pause_sec: float = 0.12,
        move_settle_sec: float = 0.2,
        lift_height_in: float = 2.0,
        lift_steps: int = 4,
        require_vision: bool = True,
    ) -> dict[str, Any]:
        """
        Pick sequence in global coordinates with staged vision verification.

        Workflow:
        1) Move to 2" above target with gripper down.
        2) Descend to 1", 0.5", 0.2", 0.0" above target, checking vision each step.
        3) Engage tool (vacuum/gripper) at pick location.
        4) Lift slowly by 2".
        """
        self.ensure_connected()
        if not self.transform_ready:
            raise RuntimeError("Robot-to-global transform is not calibrated yet.")

        target_global = self._as_xyz([x, y, z])
        target_robot = self.global_to_robot_xyz(target_global)

        if orientation_rpy is None:
            orientation = self.default_orientation_rpy
        else:
            orientation = tuple(float(v) for v in orientation_rpy)
            if len(orientation) != 3:
                raise ValueError("orientation_rpy must contain roll, pitch, yaw.")

        global_units_per_inch = self._global_units_per_inch()
        verify_tolerance_global = float(verify_tolerance_in) * global_units_per_inch
        if verify_tolerance_global < 0.0:
            raise ValueError("verify_tolerance_in must be >= 0.")

        approach_offset_global = float(approach_height_in) * global_units_per_inch
        if approach_offset_global < 0.0:
            raise ValueError("approach_height_in must be >= 0.")

        checkpoints = [float(v) for v in descent_checkpoints_in]
        if not checkpoints:
            checkpoints = [0.0]

        def _move_and_wait(expected_global_xyz: np.ndarray, stage: str) -> dict[str, Any]:
            move_result = self.move_global_xyz(
                float(expected_global_xyz[0]),
                float(expected_global_xyz[1]),
                float(expected_global_xyz[2]),
                roll=orientation[0],
                pitch=orientation[1],
                yaw=orientation[2],
                frame=frame,
            )
            if move_settle_sec > 0:
                time.sleep(float(move_settle_sec))

            verify_info = {
                "stage": stage,
                "expected_global_xyz": expected_global_xyz.astype(float).tolist(),
                "observed_global_xyz": None,
                "error_global": None,
                "error_in": None,
                "ok": None,
                "attempts": 0,
            }
            if observe_gripper_global_xyz_fn is None:
                if require_vision:
                    raise RuntimeError("observe_gripper_global_xyz_fn is required for object pick verification.")
                verify_info["ok"] = True
                return {"move": move_result, "verify": verify_info}

            last_error = None
            for attempt in range(max(1, int(verify_retries))):
                verify_info["attempts"] = attempt + 1
                if force_vision_update_fn is not None:
                    force_vision_update_fn()
                observed_raw = observe_gripper_global_xyz_fn()
                if observed_raw is None:
                    last_error = "observer_returned_none"
                    if verify_pause_sec > 0:
                        time.sleep(float(verify_pause_sec))
                    continue
                observed = self._as_xyz(observed_raw)
                err_global = float(np.linalg.norm(observed - expected_global_xyz))
                err_in = err_global / global_units_per_inch
                verify_info["observed_global_xyz"] = observed.astype(float).tolist()
                verify_info["error_global"] = err_global
                verify_info["error_in"] = err_in
                verify_info["ok"] = bool(err_global <= verify_tolerance_global)
                if verify_info["ok"]:
                    return {"move": move_result, "verify": verify_info}
                last_error = f"error {err_in:.3f}in exceeds tolerance {verify_tolerance_in:.3f}in"
                if verify_pause_sec > 0:
                    time.sleep(float(verify_pause_sec))

            if require_vision:
                raise RuntimeError(f"{stage} verification failed: {last_error}")
            return {"move": move_result, "verify": verify_info}

        sequence: list[dict[str, Any]] = []

        approach_global = target_global.copy()
        approach_global[2] += approach_offset_global
        sequence.append(_move_and_wait(approach_global, stage="approach_2in"))

        for offset_in in checkpoints:
            offset_global = float(offset_in) * global_units_per_inch
            waypoint = target_global.copy()
            waypoint[2] += offset_global
            stage = f"descent_{offset_in:.3f}in"
            sequence.append(_move_and_wait(waypoint, stage=stage))

        # Activate tool at the presumed object-contact point.
        self.grasp()

        # Lift back up slowly.
        lift_global = float(lift_height_in) * global_units_per_inch
        lift_steps = max(1, int(lift_steps))
        lift_sequence: list[dict[str, Any]] = []
        for step_idx in range(1, lift_steps + 1):
            frac = float(step_idx) / float(lift_steps)
            waypoint = target_global.copy()
            waypoint[2] += frac * lift_global
            lift_move = self.move_global_xyz(
                float(waypoint[0]),
                float(waypoint[1]),
                float(waypoint[2]),
                roll=orientation[0],
                pitch=orientation[1],
                yaw=orientation[2],
                frame=frame,
            )
            lift_sequence.append(
                {
                    "stage": f"lift_{frac * float(lift_height_in):.3f}in",
                    "move": lift_move,
                }
            )
            if move_settle_sec > 0:
                time.sleep(float(move_settle_sec))

        return {
            "target_global_xyz": target_global.astype(float).tolist(),
            "target_robot_xyz": target_robot.astype(float).tolist(),
            "orientation_rpy": [float(orientation[0]), float(orientation[1]), float(orientation[2])],
            "global_units_per_inch": float(global_units_per_inch),
            "verify_tolerance_in": float(verify_tolerance_in),
            "sequence": sequence,
            "lift_sequence": lift_sequence,
            "tool_action": "grasp",
        }

    @staticmethod
    def _fit_similarity_transform(
        robot_points: np.ndarray,
        global_points: np.ndarray,
        allow_scale: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if robot_points.shape != global_points.shape:
            raise ValueError("robot_points and global_points must have the same shape.")
        if robot_points.ndim != 2 or robot_points.shape[1] != 3:
            raise ValueError("Expected point arrays of shape (N, 3).")
        if robot_points.shape[0] < 4:
            raise ValueError("At least 4 point pairs are required to solve a stable 3D transform.")

        rp = robot_points.astype(np.float64)
        gp = global_points.astype(np.float64)
        if not np.isfinite(rp).all() or not np.isfinite(gp).all():
            raise ValueError("Point arrays must contain finite values only.")

        rp_mean = rp.mean(axis=0)
        gp_mean = gp.mean(axis=0)
        rp_centered = rp - rp_mean
        gp_centered = gp - gp_mean

        cov = (gp_centered.T @ rp_centered) / rp.shape[0]
        U, singular_vals, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt

        if allow_scale:
            var_rp = np.sum(rp_centered ** 2) / rp.shape[0]
            if var_rp <= 1e-12:
                raise ValueError("Degenerate robot point cloud; cannot solve scale.")
            s = float(np.sum(singular_vals) / var_rp)
        else:
            s = 1.0

        t = gp_mean - s * (R @ rp_mean)
        predicted = (s * (R @ rp.T)).T + t
        rmse = float(np.sqrt(np.mean(np.sum((predicted - gp) ** 2, axis=1))))
        return R, t, s, rmse

    def calibrate_from_correspondences(
        self,
        robot_points_xyz: Iterable[Iterable[float]],
        global_points_xyz: Iterable[Iterable[float]],
        allow_scale: Optional[bool] = None,
    ) -> dict[str, Any]:
        allow_scale = self.allow_scale_by_default if allow_scale is None else bool(allow_scale)

        robot_np = np.asarray(list(robot_points_xyz), dtype=np.float64)
        global_np = np.asarray(list(global_points_xyz), dtype=np.float64)
        R, t, s, rmse = self._fit_similarity_transform(robot_np, global_np, allow_scale=allow_scale)
        transform = self.set_transform(
            R,
            t,
            scale_robot_to_global=s,
            source="ned2_grid_calibration",
            rmse=rmse,
        )
        summary = {
            "num_points": int(robot_np.shape[0]),
            "allow_scale": bool(allow_scale),
            "transform": transform,
        }
        self.last_calibration_summary = summary
        return summary

    def run_grid_calibration(
        self,
        observe_global_xyz_fn: Callable[[np.ndarray, int], Optional[Iterable[float]]],
        x_values: Iterable[float],
        y_values: Iterable[float],
        z_values: Iterable[float],
        orientation_rpy: Optional[Iterable[float]] = None,
        settle_time_sec: float = 0.35,
        allow_scale: Optional[bool] = None,
        require_all_points: bool = False,
    ) -> dict[str, Any]:
        """
        Move the arm through a 3x3x3-style grid (or any provided grid) and ask
        computer vision for the global gripper position at each stop.

        observe_global_xyz_fn(robot_xyz, sample_index) should return global XYZ or None.
        """
        self.ensure_connected()
        allow_scale = self.allow_scale_by_default if allow_scale is None else bool(allow_scale)

        xs = [float(v) for v in x_values]
        ys = [float(v) for v in y_values]
        zs = [float(v) for v in z_values]
        if not xs or not ys or not zs:
            raise ValueError("x_values, y_values and z_values must all be non-empty.")

        if orientation_rpy is None:
            orientation = self.default_orientation_rpy
        else:
            orientation = tuple(float(v) for v in orientation_rpy)
            if len(orientation) != 3:
                raise ValueError("orientation_rpy must contain roll, pitch, yaw.")

        robot_points: list[np.ndarray] = []
        global_points: list[np.ndarray] = []
        samples: list[dict[str, Any]] = []

        sample_idx = 0
        for z in zs:
            for y in ys:
                for x in xs:
                    robot_xyz = np.array([x, y, z], dtype=np.float64)
                    row: dict[str, Any] = {
                        "sample_index": sample_idx,
                        "robot_xyz": robot_xyz.tolist(),
                        "global_xyz": None,
                        "ok": False,
                        "error": None,
                    }
                    sample_idx += 1
                    try:
                        self.move_robot_xyz(
                            x,
                            y,
                            z,
                            roll=orientation[0],
                            pitch=orientation[1],
                            yaw=orientation[2],
                        )
                        if settle_time_sec > 0:
                            time.sleep(float(settle_time_sec))

                        global_xyz_raw = observe_global_xyz_fn(robot_xyz.copy(), row["sample_index"])
                        if global_xyz_raw is None:
                            row["error"] = "observer_returned_none"
                            samples.append(row)
                            continue

                        global_xyz = self._as_xyz(global_xyz_raw)
                        row["global_xyz"] = global_xyz.tolist()
                        row["ok"] = True
                        samples.append(row)

                        robot_points.append(robot_xyz)
                        global_points.append(global_xyz)
                    except Exception as exc:
                        row["error"] = str(exc)
                        samples.append(row)

        expected = len(xs) * len(ys) * len(zs)
        observed = len(robot_points)
        if require_all_points and observed != expected:
            raise RuntimeError(
                f"Grid calibration expected {expected} valid points but got {observed}."
            )
        if observed < 4:
            raise RuntimeError(
                f"Grid calibration requires at least 4 valid correspondences; got {observed}."
            )

        calibration = self.calibrate_from_correspondences(
            robot_points_xyz=np.asarray(robot_points, dtype=np.float64),
            global_points_xyz=np.asarray(global_points, dtype=np.float64),
            allow_scale=allow_scale,
        )
        result = {
            "expected_grid_points": int(expected),
            "observed_grid_points": int(observed),
            "samples": samples,
            "calibration": calibration,
        }
        self.last_calibration_summary = result
        return result
