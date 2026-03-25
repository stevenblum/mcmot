#!/usr/bin/env python3

from argparse import ArgumentParser
import base64
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Output-size knobs (easy to tweak at the top of this file).
DEFAULT_MAX_WIDTH = 700
DEFAULT_JPEG_QUALITY = 30
DEFAULT_KP_CONF = 0.5
# Supervision has no built-in skeleton for 5 keypoints, so define one explicitly.
# Adjust this order if your keypoint indexing uses a different topology.
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
            return best_path

    raise FileNotFoundError(f"No best.pt found under: {runs_dir}")


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_for_display(image_bgr, max_width: int):
    if max_width <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr

    scale = max_width / float(width)
    return cv2.resize(
        image_bgr,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def image_to_data_uri(image_bgr, jpeg_quality: int) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG for Plotly report")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


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


def resolve_keypoint_annotator_class(sv):
    keypoint_annotator_cls = getattr(sv, "KeyPointAnnotator", None)
    if keypoint_annotator_cls is not None:
        return keypoint_annotator_cls
    return sv.VertexAnnotator


def build_report(
    model,
    image_paths: list[Path],
    output_path: Path,
    pred_conf: float,
    kp_conf: float,
    max_width: int,
    jpeg_quality: int,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required for HTML output. Install with: pip install plotly"
        ) from exc

    try:
        import supervision as sv
    except ImportError as exc:
        raise SystemExit(
            "Supervision is required for keypoint parsing/annotation. Install with: pip install supervision"
        ) from exc

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

    titles = [p.name for p in image_paths]
    n_rows = len(image_paths)
    if n_rows <= 1:
        vertical_spacing = 0.0
    else:
        # Keep spacing small so each row retains visible height.
        max_spacing = max(0.0, (1.0 / (n_rows - 1)) - 1e-6)
        vertical_spacing = min(0.003, max_spacing)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=vertical_spacing,
    )

    for row_idx, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        # 1) Detect with YOLO pose model.
        results = model.predict(source=str(image_path), conf=pred_conf, verbose=False)
        # 2) Convert Ultralytics result to Supervision KeyPoints.
        key_points = sv.KeyPoints.from_ultralytics(results[0])
        key_points = filter_keypoints_by_confidence(sv, key_points, min_conf=kp_conf)

        # 3) Annotate using Supervision edge/keypoint annotators.
        annotated = image_bgr.copy()
        annotated = edge_annotator.annotate(scene=annotated, key_points=key_points)
        annotated = keypoint_annotator.annotate(scene=annotated, key_points=key_points)

        display_bgr = resize_for_display(annotated, max_width=max_width)
        display_height, display_width = display_bgr.shape[:2]
        fig.add_trace(
            go.Image(source=image_to_data_uri(display_bgr, jpeg_quality=jpeg_quality)),
            row=row_idx,
            col=1,
        )
        fig.update_xaxes(visible=False, range=[0, display_width], row=row_idx, col=1)
        fig.update_yaxes(visible=False, range=[display_height, 0], row=row_idx, col=1)

    fig.update_layout(
        title_text="Pose Predictions (Supervision KeyPoints + Edge Annotators)",
        height=max(600, 320 * n_rows),
        width=max(1000, max_width + 250),
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    default_images_dir = parent_dir / "ned2_pose_dataset" / "train" / "images"
    default_output = parent_dir / "ned2_pose_viz.html"

    parser = ArgumentParser(description="Visualize all pose images in a single Plotly HTML report.")
    parser.add_argument("--model", type=Path, default=None, help="Path to model .pt (default: latest run)")
    parser.add_argument("--images-dir", type=Path, default=default_images_dir, help="Directory of images")
    parser.add_argument("--output", type=Path, default=default_output, help="Output HTML path")
    parser.add_argument("--pred-conf", type=float, default=0.25, help="YOLO prediction confidence threshold")
    parser.add_argument(
        "--kp-conf",
        type=float,
        default=DEFAULT_KP_CONF,
        help="Minimum keypoint confidence to display",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help="Max display width per image in HTML",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for embedded images (1-100)",
    )
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else (parent_dir / args.output)

    model_path = args.model if args.model is not None else find_most_recent_run(parent_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if not (0.0 <= args.kp_conf <= 1.0):
        raise ValueError("--kp-conf must be between 0 and 1")

    image_paths = collect_images(args.images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {args.images_dir}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required to run predictions. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    build_report(
        model=model,
        image_paths=image_paths,
        output_path=output_path,
        pred_conf=args.pred_conf,
        kp_conf=args.kp_conf,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"Wrote Plotly report with {len(image_paths)} images: {output_path}")


if __name__ == "__main__":
    main()
