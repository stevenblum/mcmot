#python3 ned2_pose/extract_frames.py --video ned2_pose/Vendo_Agnostic_Testbed_6parts.mp4 --interval 1.0

from argparse import ArgumentParser
from pathlib import Path

import cv2


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_video = script_dir / "Vendo_Agnostic_Testbed_6parts.mp4"
    out_dir = script_dir.parent / "images"

    parser = ArgumentParser(description="Save MP4 frames every N seconds.")
    parser.add_argument("--video", type=Path, default=default_video, help="Path to .mp4")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames")
    args = parser.parse_args()

    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * args.interval)))
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            ts = frame_idx / fps
            name = f"{args.video.stem}_{saved:05d}_{ts:08.2f}s.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frame(s) to: {out_dir}")


if __name__ == "__main__":
    main()
