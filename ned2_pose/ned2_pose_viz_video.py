# python3 ned2_pose/ned2_pose_viz_video.py --input-video Vendor_Agnostic_Testbed_6parts.mp4 --output-width-res 800 --output-x-speed 4
# python3 ned2_pose/ned2_pose_viz_video.py --input-video best_video_normal_segment_006.mp4 --output-width-res 800 --output-x-speed 4
#!/usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2

# Annotation knobs.
DEFAULT_PRED_CONF = 0.25
DEFAULT_KP_CONF = 0.5
DEFAULT_TRACKER = "botsort.yaml"
DEFAULT_OUTPUT_WIDTH_RES = 0
DEFAULT_OUTPUT_X_SPEED = 1
DEFAULT_KEYPOINT_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5)]
DEFAULT_KEYPOINT_RADIUS = 11
DEFAULT_EDGE_THICKNESS = 6


def find_most_recent_run(parent_dir: Path) -> Path:
    runs_dir = parent_dir / "ned2_pose_train_results"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No training runs directory found: {runs_dir}")

    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run")]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        best_path = run_dir / "weights" / "best.pt"
        if best_path.exists():
            model_saved_at = datetime.fromtimestamp(best_path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print(f"Using model: {best_path}")
            print(f"Model saved at: {model_saved_at}")
            return best_path

    raise FileNotFoundError(f"No best.pt found under: {runs_dir}")


def resolve_keypoint_annotator_class(sv):
    keypoint_annotator_cls = getattr(sv, "KeyPointAnnotator", None)
    if keypoint_annotator_cls is not None:
        return keypoint_annotator_cls
    return sv.VertexAnnotator


def filter_keypoints_by_confidence(sv, key_points, min_conf: float):
    if len(key_points) == 0 or key_points.confidence is None:
        return key_points

    confidence_mask = key_points.confidence >= min_conf
    filtered_xy = key_points.xy.copy()
    filtered_xy[~confidence_mask] = 0
    return sv.KeyPoints(
        xy=filtered_xy,
        class_id=key_points.class_id,
        confidence=key_points.confidence,
        data=key_points.data,
    )


def process_video(
    model,
    input_video: Path,
    output_video: Path,
    pred_conf: float,
    kp_conf: float,
    tracker: str,
    output_width_res: int,
    output_x_speed: int,
) -> None:
    try:
        import supervision as sv
    except ImportError as exc:
        raise SystemExit(
            "Supervision is required for keypoint parsing/annotation. Install with: pip install supervision"
        ) from exc

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("tqdm is required for progress display. Install with: pip install tqdm") from exc

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None
    cap.release()

    if output_width_res > 0:
        out_width = output_width_res
        out_height = int(round(height * (out_width / width)))
    else:
        out_width = width
        out_height = height

    # Many encoders expect even dimensions.
    out_width = max(2, out_width - (out_width % 2))
    out_height = max(2, out_height - (out_height % 2))
    print(f"Output resolution: {out_width}x{out_height}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, out_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_video}")

    keypoint_annotator_cls = resolve_keypoint_annotator_class(sv)
    try:
        keypoint_annotator = keypoint_annotator_cls(
            color=sv.Color.GREEN,
            radius=DEFAULT_KEYPOINT_RADIUS,
        )
    except TypeError:
        keypoint_annotator = keypoint_annotator_cls()
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.RED,
        thickness=DEFAULT_EDGE_THICKNESS,
        edges=DEFAULT_KEYPOINT_EDGES,
    )

    pbar = tqdm(total=total_frames, desc="Rendering video", unit="frame")
    frame_idx = 0
    saved_frames = 0
    try:
        # Use Ultralytics tracker (BoT-SORT by default) across the full video stream.
        track_stream = model.track(
            source=str(input_video),
            conf=pred_conf,
            tracker=tracker,
            persist=True,
            stream=True,
            verbose=False,
        )

        for result in track_stream:
            frame = result.orig_img
            if frame is None:
                frame_idx += 1
                pbar.update(1)
                continue

            # Track on every frame, but only save every Nth frame.
            if frame_idx % output_x_speed == 0:
                # 1) Detect + track
                # 2) Convert
                key_points = sv.KeyPoints.from_ultralytics(result)
                key_points = filter_keypoints_by_confidence(sv, key_points, min_conf=kp_conf)
                # 3) Annotate
                annotated = edge_annotator.annotate(scene=frame.copy(), key_points=key_points)
                annotated = keypoint_annotator.annotate(scene=annotated, key_points=key_points)

                if annotated.shape[1] != out_width or annotated.shape[0] != out_height:
                    annotated = cv2.resize(annotated, (out_width, out_height), interpolation=cv2.INTER_AREA)
                writer.write(annotated)
                saved_frames += 1

            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        writer.release()
        print(f"Saved frames: {saved_frames}")
        print(f"Frame save interval (--output-x-speed): {output_x_speed}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    default_input_video = script_dir / "best_video_normal_segment_006.mp4"

    parser = ArgumentParser(description="Run pose inference on a video and save annotated output.")
    parser.add_argument(
        "--input-video",
        type=Path,
        default=default_input_video,
        help="Input .mp4 video path",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Output .mp4 path (optional). If omitted, uses input name with _annotated suffix.",
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to model .pt (default: latest run)")
    parser.add_argument(
        "--pred-conf",
        type=float,
        default=DEFAULT_PRED_CONF,
        help="YOLO prediction confidence threshold",
    )
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=DEFAULT_KP_CONF,
        help="Minimum keypoint confidence to display",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER,
        help="Ultralytics tracker config, e.g. botsort.yaml or bytetrack.yaml",
    )
    parser.add_argument(
        "--output-width-res",
        type=int,
        default=DEFAULT_OUTPUT_WIDTH_RES,
        help="Output width in pixels. 0 keeps original width. Height is auto-scaled.",
    )
    parser.add_argument(
        "--output-x-speed",
        type=int,
        default=DEFAULT_OUTPUT_X_SPEED,
        help="Save every Nth frame (2=every other frame, 3=every third frame, etc.)",
    )
    args = parser.parse_args()

    input_video = args.input_video if args.input_video.is_absolute() else (script_dir / args.input_video)
    if args.output_video is None:
        output_video = input_video.with_name(f"{input_video.stem}_annotated{input_video.suffix}")
    else:
        output_video = (
            args.output_video if args.output_video.is_absolute() else (script_dir / args.output_video)
        )
    model_path = args.model if args.model is not None else find_most_recent_run(parent_dir)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not (0.0 <= args.kp_conf <= 1.0):
        raise ValueError("--kp-conf must be between 0 and 1")
    if args.output_width_res < 0:
        raise ValueError("--output-width-res must be >= 0")
    if args.output_x_speed < 1:
        raise ValueError("--output-x-speed must be >= 1")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required to run predictions. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    process_video(
        model=model,
        input_video=input_video,
        output_video=output_video,
        pred_conf=args.pred_conf,
        kp_conf=args.kp_conf,
        tracker=args.tracker,
        output_width_res=args.output_width_res,
        output_x_speed=args.output_x_speed,
    )
    print(f"Saved annotated video: {output_video}")


if __name__ == "__main__":
    main()
