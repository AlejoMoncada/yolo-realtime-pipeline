"""Export YOLO weights to a fixed-input inference format; no model work occurs on --help."""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path


def export_model(weights: str, imgsz: int, export_format: str = "onnx", output_dir: str | Path = ".") -> Path:
    """Export at a static input size and name the file <stem>-<imgsz>.<format> so resolutions never overwrite each other."""
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    YOLO = importlib.import_module("ultralytics").YOLO
    produced = Path(YOLO(weights).export(format=export_format, imgsz=imgsz, dynamic=False, simplify=True))
    target = Path(output_dir) / f"{Path(weights).stem}-{imgsz}{produced.suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != target.resolve():
        shutil.move(str(produced), target)
    return target


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export YOLO weights for faster inference (static input size, one file per resolution).")
    parser.add_argument("weights", help="source weights, for example yolo11n.pt")
    parser.add_argument("--imgsz", type=int, required=True, help="fixed inference resolution; pass the same value as --imgsz/--optimized-imgsz when running")
    parser.add_argument("--format", default="onnx", help="Ultralytics export format (default: onnx)")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args(argv)
    if args.imgsz <= 0:
        parser.error("imgsz must be positive")
    try:
        target = export_model(args.weights, args.imgsz, args.format, args.output_dir)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        detail = f"required runtime dependency unavailable ({exc}); install project dependencies from requirements.txt" if isinstance(exc, ImportError) else str(exc)
        print(f"export error: {detail}", file=sys.stderr)
        return 1
    print(f"Exported: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
