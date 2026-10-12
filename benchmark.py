"""Controlled local-video sync/threaded benchmark; no model work occurs on --help."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from vision_pipeline import PipelineConfig, export_run, importlib

Runner = Callable[[PipelineConfig], dict[str, Any]]


def _source_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"resolved_path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _source_frame_count(path: Path) -> int:
    cv2 = importlib.import_module("cv2")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open local video {str(path)!r}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError("source does not report a positive frame count; controlled comparison requires a complete local video")
        return count
    finally:
        capture.release()


def _validate_result(result: dict[str, Any], expected_count: int, expected_ids: list[int] | None) -> list[int]:
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark run produced no frames; empty video is not a valid comparison")
    frame_ids = [row.get("frame_id") for row in rows]
    wanted_ids = list(range(expected_count))
    if frame_ids != wanted_ids:
        raise ValueError(f"benchmark run was partial or non-contiguous: expected {expected_count} frame IDs, received {frame_ids[:8]}")
    if expected_ids is not None and frame_ids != expected_ids:
        raise ValueError("benchmark frame IDs differ between runs")
    summary = result.get("summary", {})
    if summary.get("frames") != expected_count or summary.get("stop_reason") != "eof":
        raise ValueError("benchmark run did not complete the selected source frame range")
    return frame_ids


def _case_summary(runs: list[dict[str, Any]], base_fps: float | None) -> dict[str, Any]:
    throughputs = [float(run["summary"]["annotated_throughput_fps"]) for run in runs]
    latencies = [float(run["summary"]["latency_end_to_end_ms"]["p95"]) for run in runs]
    median_fps = statistics.median(throughputs)
    median_p95 = statistics.median(latencies)
    return {
        "runs": len(runs),
        "per_run_throughput_fps": throughputs,
        "median_throughput_fps": median_fps,
        "per_run_end_to_end_p95_ms": latencies,
        "median_end_to_end_p95_ms": median_p95,
        "speedup_vs_sync": median_fps / base_fps if base_fps and base_fps > 0 else None,
        "throughput_change_percent_vs_sync": ((median_fps / base_fps) - 1) * 100 if base_fps and base_fps > 0 else None,
        "definitions": {
            "throughput": "unique completed frames divided by each run's measured elapsed seconds; report the median across runs",
            "end_to_end_p95": "linear-interpolated per-frame p95 from source-read start through annotation/display, including queue wait; report the median of run p95 values",
            "speedup": "case median throughput divided by sync median throughput",
            "throughput_change_percent": "(case median throughput / sync median throughput - 1) * 100",
        },
    }


def run_benchmark(
    source: str | Path,
    *,
    weights: str = "yolo11n.pt",
    device: str = "",
    conf: float = 0.25,
    imgsz: int = 640,
    warmup: int = 2,
    max_frames: int | None = None,
    queue_size: int = 2,
    repeats: int = 3,
    output_dir: str | Path = "benchmark-results",
    runner: Runner | None = None,
    frame_count_reader: Callable[[Path], int] | None = None,
    optimized_imgsz: int | None = None,
    optimized_weights: str | None = None,
    realtime: bool = False,
) -> dict[str, Any]:
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"source must be an existing local video file: {source!r}")
    if repeats <= 0 or imgsz <= 0 or warmup < 0 or queue_size <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("repeats, imgsz and queue-size must be positive; warmup non-negative; max-frames positive")
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("conf must be finite and in [0, 1]")
    if not weights.strip():
        raise ValueError("weights must be non-empty")
    if optimized_weights is not None and (optimized_imgsz is None or not optimized_weights.strip()):
        raise ValueError("optimized-weights requires optimized-imgsz and must be non-empty")
    run_real = runner
    if run_real is None:
        from vision_pipeline import run_real as real_adapter
        run_real = real_adapter
    identity = _source_identity(path)
    available_frames = (frame_count_reader or _source_frame_count)(path)
    expected_count = min(available_frames, max_frames) if max_frames is not None else available_frames
    if expected_count <= 0:
        raise ValueError("selected source frame range is empty")

    if optimized_imgsz is not None and optimized_imgsz <= 0:
        raise ValueError("optimized-imgsz must be positive")
    cases: list[tuple[str, int, str, str]] = [("sync", imgsz, "sync", weights), ("threaded", imgsz, "threaded", weights)]
    if optimized_imgsz is not None:
        cases.append(("threaded_optimized_imgsz", optimized_imgsz, "threaded", optimized_weights or weights))
    # A unique child directory makes reruns independent even when output_dir is reused.
    root = Path(output_dir) / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    case_runs: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _, _ in cases}
    order: list[dict[str, Any]] = []
    expected_ids: list[int] | None = None
    run_index = 0
    for repeat in range(repeats):
        rotated = cases[repeat % len(cases):] + cases[:repeat % len(cases)]
        for name, resolution, mode, case_weights in rotated:
            config = PipelineConfig(str(path.resolve()), case_weights, mode, resolution, device, conf, warmup, max_frames, False, str(root), queue_size, realtime)
            result = run_real(config)
            ids = _validate_result(result, expected_count, expected_ids)
            if expected_ids is None:
                expected_ids = ids
            run_dir = root / f"{run_index:02d}-{name}-repeat-{repeat + 1}"
            run_dir.mkdir()
            result["summary"]["benchmark"] = {"case": name, "repeat": repeat + 1, "order_index": run_index, "imgsz": resolution, "mode": mode, "weights": case_weights}
            csv_path, json_path = export_run(result, run_dir)
            case_runs[name].append({"summary": result["summary"], "csv": str(csv_path.relative_to(root)), "json": str(json_path.relative_to(root))})
            order.append({"order_index": run_index, "repeat": repeat + 1, "case": name})
            run_index += 1
    baseline = _case_summary(case_runs["sync"], None)
    comparison: dict[str, Any] = {
        "schema_version": 1,
        "input": identity,
        "selected_frame_ids": expected_ids,
        "selected_frame_count": expected_count,
        "parameters": {"weights": weights, "device": device or "auto", "conf": conf, "imgsz": imgsz, "warmup": warmup, "max_frames": max_frames, "queue_size": queue_size, "repeats": repeats, "optimized_weights": optimized_weights, "realtime": realtime},
        "environment": {
            "hardware": case_runs["sync"][0]["summary"].get("hardware"),
            "versions": case_runs["sync"][0]["summary"].get("versions"),
        },
        "run_order": order,
        "cases": {},
        "resolution_notice": "The optional optimized-imgsz case changes inference resolution and does not establish predictive-quality parity; no ground-truth quality evaluation is performed.",
        "concurrency_notice": "Sync versus threaded uses identical weights, device, confidence, resolution and selected source frame IDs.",
    }
    for name, resolution, mode, case_weights in cases:
        summary = _case_summary(case_runs[name], baseline["median_throughput_fps"])
        comparison["cases"][name] = {"imgsz": resolution, "mode": mode, "weights": case_weights, **summary, "artifacts": [{"csv": run["csv"], "summary_json": run["json"]} for run in case_runs[name]]}
    with (root / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return comparison | {"output_directory": str(root)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare sync and threaded YOLO runs on the same complete local-video frame range.")
    parser.add_argument("local_video", help="existing local video file (webcams and streams are not accepted)")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--device", default="", help="auto, cpu, mps, cuda, or CUDA index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--optimized-imgsz", type=int, help="optional separate threaded run at a different resolution")
    parser.add_argument("--optimized-weights", help="weights for the optimized case only, e.g. an ONNX export at --optimized-imgsz (see export_model.py)")
    parser.add_argument("--realtime", action="store_true", help="pace every case at the source FPS to emulate a live camera; latency then includes any backlog")
    parser.add_argument("--output-dir", default="benchmark-results")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    path = Path(args.local_video).expanduser()
    if not path.is_file():
        parser.error("local_video must be an existing local file; webcams and streams are not supported")
    if (args.imgsz <= 0 or args.warmup < 0 or args.queue_size <= 0 or args.repeats <= 0
            or (args.max_frames is not None and args.max_frames <= 0)
            or (args.optimized_imgsz is not None and args.optimized_imgsz <= 0)
            or not math.isfinite(args.conf) or not 0 <= args.conf <= 1
            or not args.weights.strip()
            or (args.optimized_weights is not None and (args.optimized_imgsz is None or not args.optimized_weights.strip()))):
        parser.error("imgsz, queue-size and repeats must be positive; warmup non-negative; max-frames positive; conf finite in [0, 1]; weights non-empty; optimized-weights requires optimized-imgsz")
    try:
        report = run_benchmark(path, weights=args.weights, device=args.device, conf=args.conf, imgsz=args.imgsz,
                               warmup=args.warmup, max_frames=args.max_frames, queue_size=args.queue_size,
                               repeats=args.repeats, output_dir=args.output_dir, optimized_imgsz=args.optimized_imgsz,
                               optimized_weights=args.optimized_weights, realtime=args.realtime)
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        detail = f"required runtime dependency unavailable ({exc}); install project dependencies from requirements.txt" if isinstance(exc, ImportError) else str(exc)
        print(f"benchmark error: {detail}", file=sys.stderr)
        return 1
    for name, case in report["cases"].items():
        print(f"{name}: median throughput {case['median_throughput_fps']:.3f} FPS; median run p95 E2E {case['median_end_to_end_p95_ms']:.3f} ms; speedup vs sync {case['speedup_vs_sync']:.3f}x")
    print(f"Results: {report['output_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
