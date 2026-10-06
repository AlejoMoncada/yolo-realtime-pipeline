"""Generate a small synthetic OpenCV clip for pipeline wiring checks only."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a labelled moving-shape video; this is not model-quality or performance evidence.")
    parser.add_argument("output", nargs="?", default="demo-video.avi")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args(argv)
    if args.frames <= 0 or args.width < 64 or args.height < 64 or args.fps <= 0:
        parser.error("frames and fps must be positive; width and height must be at least 64")
    from importlib import import_module
    cv2 = import_module("cv2")
    np = import_module("numpy")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"could not create video at {output}")
    try:
        for frame_id in range(args.frames):
            frame = np.full((args.height, args.width, 3), (24, 27, 34), dtype=np.uint8)
            x = 24 + (frame_id * 7) % max(1, args.width - 48)
            y = args.height // 2 + int((args.height // 4) * np.sin(frame_id * 0.16))
            cv2.rectangle(frame, (x - 14, y - 14), (x + 14, y + 14), (60, 210, 255), -1)
            cv2.circle(frame, (args.width - x, args.height - y), 12, (90, 230, 120), -1)
            cv2.putText(frame, f"synthetic wiring demo  frame {frame_id:03d}", (8, args.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created {args.frames} synthetic frames at {args.fps:g} FPS: {output}")
    print("Synthetic clip is a wiring example, not accuracy or realistic YOLO-performance evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
