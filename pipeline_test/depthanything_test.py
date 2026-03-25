#!/usr/bin/env python3
"""
Single-camera Depth Anything test (ONNX Runtime).

Requirements:
- onnxruntime
- OpenCV
- A Depth Anything ONNX model file

Windows:
- Camera
- Depth Anything

Controls:
- q or ESC: quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ort = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
VENV_ROOT = PROJECT_ROOT / ".venv"


def ensure_project_venv() -> None:
    if not VENV_PYTHON.exists():
        return
    current_prefix = os.path.realpath(sys.prefix)
    expected_prefix = os.path.realpath(str(VENV_ROOT))
    if current_prefix != expected_prefix:
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])


class DepthAnythingONNX:
    def __init__(self, model_path: Path, input_size: int, provider: str) -> None:
        if ort is None:  # pragma: no cover - runtime dependency check
            raise RuntimeError(
                "onnxruntime is required for depthanything_test.py. "
                "Install it in your environment and retry."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Depth Anything ONNX model not found: {model_path}\n"
                "Pass --model with a valid .onnx file path."
            )

        providers = self._resolve_providers(provider)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.active_provider = self.session.get_providers()[0]

        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        self.input_name = input_meta.name
        self.output_name = output_meta.name

        self.in_h = int(input_size)
        self.in_w = int(input_size)
        if len(input_meta.shape) == 4:
            h = input_meta.shape[2]
            w = input_meta.shape[3]
            if isinstance(h, int) and h > 0:
                self.in_h = h
            if isinstance(w, int) and w > 0:
                self.in_w = w

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _resolve_providers(provider: str) -> list[str]:
        available = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider requested but not available in onnxruntime."
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _to_depth_map(output: np.ndarray) -> np.ndarray:
        depth = np.asarray(output)
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[2] == 1:
                depth = depth[:, :, 0]
            else:
                depth = depth[0]
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected depth output shape: {np.asarray(output).shape}")
        return depth.astype(np.float32)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = cv2.resize(image, (self.in_w, self.in_h), interpolation=cv2.INTER_CUBIC)
        image = (image - self.mean) / self.std
        tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32)

        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        depth = self._to_depth_map(output)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
        return depth


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if np.count_nonzero(finite) < 10:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d = depth.copy()
    d[~finite] = np.nan
    lo = float(np.nanpercentile(d, 2))
    hi = float(np.nanpercentile(d, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(d))
        hi = float(np.nanmax(d))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

    d = np.clip(d, lo, hi)
    d = ((d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth Anything single-camera test.")
    parser.add_argument("--camera-port", type=int, default=4, help="Camera device id.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "pipeline_test" / "models" / "depth_anything_v2_vits.onnx"),
        help="Path to Depth Anything ONNX model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Square model input size for dynamic-input models.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional capture height.")
    parser.add_argument(
        "--print-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between FPS prints.",
    )
    return parser.parse_args()


def main() -> None:
    global ort
    ensure_project_venv()
    args = parse_args()

    try:
        import onnxruntime as _ort
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "onnxruntime is required for depthanything_test.py. "
            "Install it in your environment and retry."
        ) from exc
    ort = _ort

    model = DepthAnythingONNX(Path(args.model), args.input_size, args.provider)
    print(f"Loaded model: {args.model}")
    print(f"ONNX Runtime provider: {model.active_provider}")

    cap = cv2.VideoCapture(args.camera_port)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera on port {args.camera_port}.")
    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps_ema = None
    last_print = time.time()

    try:
        while True:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            infer_start = time.time()
            depth = model.infer(frame)
            infer_ms = (time.time() - infer_start) * 1000.0
            depth_vis = depth_to_colormap(depth)

            loop_dt = max(time.time() - loop_start, 1e-6)
            inst_fps = 1.0 / loop_dt
            fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)

            label = f"FPS: {fps_ema:.1f} | Infer: {infer_ms:.1f} ms"
            cv2.putText(
                frame,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_vis,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Camera", frame)
            cv2.imshow("Depth Anything", depth_vis)

            now = time.time()
            if now - last_print >= args.print_interval_sec:
                print(f"{label}")
                last_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
