"""Synchronous, instrumented video inference baseline."""

from __future__ import annotations

import contextlib
import csv
import importlib
import json
import math
import os
import platform
import queue
import statistics
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineConfig:
    source: str
    weights: str = "yolo11n.pt"
    mode: str = "sync"
    imgsz: int = 640
    device: str = ""
    conf: float = 0.25
    warmup: int = 0
    max_frames: int | None = None
    show: bool = False
    output_dir: str = "results"
    queue_size: int = 2


def _validate_config(config: PipelineConfig, mode: str) -> None:
    if config.mode != mode:
        raise ValueError(f"only --mode {mode} is valid for run_{mode}" if mode == "sync" else "run_threaded requires mode='threaded'")
    if config.imgsz <= 0 or config.warmup < 0 or config.queue_size <= 0 or not math.isfinite(config.conf) or not 0 <= config.conf <= 1:
        message = "imgsz must be positive, warmup non-negative, and conf in [0, 1]" if mode == "sync" else "imgsz and queue_size must be positive, warmup non-negative, and conf in [0, 1]"
        raise ValueError(message)
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("max_frames must be positive")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _sync_device(device: str) -> None:
    """Synchronize async accelerator work when the selected backend supports it."""
    normalized = device.lower()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    cuda_selected = normalized.startswith("cuda") or normalized.isdecimal()
    auto_selected = normalized in ("", "auto")
    mps_available = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if (cuda_selected or auto_selected) and torch.cuda.is_available():
        selected_device = f"cuda:{normalized}" if normalized.isdecimal() else (device if normalized.startswith("cuda") else None)
        torch.cuda.synchronize(selected_device)
    elif hasattr(torch, "mps") and ((normalized.startswith("mps") and mps_available) or (auto_selected and mps_available)):
        torch.mps.synchronize()


def _detection_count(results: Any) -> int:
    if results is None:
        return 0
    if not isinstance(results, (list, tuple)):
        results = [results]
    total = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            with contextlib.suppress(TypeError):
                total += len(boxes)
    return total


def _result_speed(results: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(results, (list, tuple)) or not results:
        return None, None, None
    speed = getattr(results[0], "speed", None)
    if not isinstance(speed, dict):
        return None, None, None
    return (
        speed.get("preprocess"),
        speed.get("inference"),
        speed.get("postprocess"),
    )


def _percentile_summary(values: list[float]) -> dict[str, float | None]:
    return _stats(values)


def run_sync(
    config: PipelineConfig,
    capture: Any,
    predictor: Callable[[Any], Any],
    renderer: Callable[[Any, Any], Any],
    *,
    source_fps: float | None = None,
    model_name: str | None = None,
    hardware: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    wall_clock: Callable[[], float] = time.time,
    synchronizer: Callable[[str], None] = _sync_device,
    display: Callable[[str, Any], None] | None = None,
    wait_key: Callable[[int], int] | None = None,
    resource_sampler: Callable[[], dict[str, float | int | None]] | None = None,
) -> dict[str, Any]:
    """Run capture→predict→render serially; dependencies are injectable for tests."""
    _validate_config(config, "sync")
    if config.show and (display is None or wait_key is None):
        raise ValueError("display and wait_key callbacks are required when show is enabled")

    rows: list[dict[str, Any]] = []
    started: float | None = None
    measured_end: float | None = None
    stop_reason = "eof"
    try:
        # Dummy warm-up happens before the first source read, so it consumes no video.
        if config.warmup:
            np = importlib.import_module("numpy")
            dummy = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            for _ in range(config.warmup):
                synchronizer(config.device)
                predictor(dummy)
                synchronizer(config.device)
        while config.max_frames is None or len(rows) < config.max_frames:
            read_started = clock()
            if started is None:
                started = read_started
            captured = capture.read()
            read_finished = clock()
            if not isinstance(captured, tuple) or len(captured) != 2:
                raise RuntimeError("capture.read() must return (success, frame)")
            ok, frame = captured
            if not ok or frame is None:
                stop_reason = "eof"
                break

            frame_id = len(rows)
            capture_time = wall_clock()
            predict_started = clock()
            synchronizer(config.device)
            predictions = predictor(frame)
            synchronizer(config.device)
            predict_wall_ms = (clock() - predict_started) * 1000
            preprocess_ms, inference_ms, postprocess_ms = _result_speed(predictions)

            annotate_started = clock()
            annotated = renderer(frame, predictions)
            annotation_finished = clock()
            display_started = annotation_finished
            if config.show:
                assert display is not None and wait_key is not None
                display_started = clock()
                display("Synchronous vision pipeline", annotated)
                key = wait_key(1)
                rendered_at = clock()
                if key & 0xFF == ord("q"):
                    stop_reason = "user_quit"
            else:
                rendered_at = annotation_finished
            resource = resource_sampler() if resource_sampler else {"process_cpu_percent": None, "rss_bytes": None}
            rows.append({
                "frame_id": frame_id,
                "capture_time_unix_s": capture_time,
                "capture_ms": (read_finished - read_started) * 1000,
                "queue_wait_ms": 0.0,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "predict_wall_ms": predict_wall_ms,
                "annotation_ms": (annotation_finished - annotate_started) * 1000,
                "display_ms": (rendered_at - display_started) * 1000 if config.show else None,
                "annotation_display_ms": (rendered_at - annotate_started) * 1000,
                "end_to_end_ms": (rendered_at - read_started) * 1000,
                "completion_offset_s": rendered_at - started,
                "detection_count": _detection_count(predictions),
                "process_cpu_percent": resource.get("process_cpu_percent"),
                "rss_bytes": resource.get("rss_bytes"),
            })
            if stop_reason == "user_quit":
                break
        measured_end = clock()
    except Exception as exc:
        raise RuntimeError(f"synchronous pipeline failed: {exc}") from exc
    finally:
        try:
            capture.release()
        finally:
            if config.show:
                with contextlib.suppress(ImportError):
                    importlib.import_module("cv2").destroyAllWindows()

    elapsed = max((measured_end - started), 0.0) if started is not None and measured_end is not None else 0.0
    return _assemble_run(config, rows, elapsed, stop_reason, source_fps, model_name, hardware)


def _assemble_run(
    config: PipelineConfig,
    rows: list[dict[str, Any]],
    elapsed: float,
    stop_reason: str,
    source_fps: float | None,
    model_name: str | None,
    hardware: dict[str, Any] | None,
) -> dict[str, Any]:
    offsets = [float(row["completion_offset_s"]) for row in rows]
    windows = [sum(start <= offset < start + 1.0 for offset in offsets) for start in range(int(elapsed // 1.0))]
    annotated_fps = len(rows) / elapsed if elapsed > 0 else None
    stage_names = ("capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "predict_wall_ms", "queue_wait_ms", "annotation_ms", "display_ms", "annotation_display_ms", "end_to_end_ms")
    summary = {
        "config": asdict(config),
        "versions": _versions(),
        "hardware": hardware,
        "model": model_name or config.weights,
        "frames": len(rows),
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "annotated_throughput_fps": annotated_fps,
        "display_fps": annotated_fps if config.show else None,
        "source_fps": source_fps,
        "latency_end_to_end_ms": _percentile_summary([float(row["end_to_end_ms"]) for row in rows]),
        "predict_wall_ms": _percentile_summary([float(row["predict_wall_ms"]) for row in rows]),
        "stage_timings_ms": {name: _percentile_summary([float(row[name]) for row in rows if row.get(name) is not None]) for name in stage_names},
        "process_cpu_percent_semantics": "psutil process utilization sampled after each frame; first sample may be 0; null when unavailable",
        "rss_bytes_semantics": "process resident set size sampled after each frame; null when unavailable",
        "stability_window_s": 1,
        "complete_stability_windows": windows,
        "stability_windows_exclude_trailing_partial": True,
        "timing_duration": "elapsed_s spans first measured source-read start through stop/completion; excludes model setup, warm-up, worker shutdown, and cleanup",
    }
    return {"rows": rows, "summary": summary}


def run_threaded(
    config: PipelineConfig,
    capture: Any,
    predictor: Callable[[Any], Any],
    renderer: Callable[[Any, Any], Any],
    *,
    source_fps: float | None = None,
    model_name: str | None = None,
    hardware: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    wall_clock: Callable[[], float] = time.time,
    synchronizer: Callable[[str], None] = _sync_device,
    display: Callable[[str, Any], None] | None = None,
    wait_key: Callable[[int], int] | None = None,
    resource_sampler: Callable[[], dict[str, float | int | None]] | None = None,
    join_timeout: float = 1.0,
) -> dict[str, Any]:
    """Overlap owned capture and single-caller inference; render/display on caller thread."""
    _validate_config(config, "threaded")
    if config.show and (display is None or wait_key is None):
        raise ValueError("display and wait_key callbacks are required when show is enabled")

    frames: queue.Queue[Any] = queue.Queue(config.queue_size)
    results: queue.Queue[Any] = queue.Queue(config.queue_size)
    cancelled = threading.Event()
    errors: list[tuple[str, BaseException]] = []
    errors_lock = threading.Lock()
    eof = object()
    done = object()
    rows: list[dict[str, Any]] = []
    stop_reason = "eof"
    started: float | None = None
    measured_end: float | None = None
    first_read_start: list[float] = []
    first_read_started = threading.Event()
    capture_thread: threading.Thread | None = None
    inference_thread: threading.Thread | None = None
    capture_owned_by_worker = False

    def fail(stage: str, exc: BaseException) -> None:
        with errors_lock:
            errors.append((stage, exc))
        cancelled.set()

    def put_bounded(target: queue.Queue[Any], item: Any) -> bool:
        while not cancelled.is_set():
            try:
                target.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def capture_worker() -> None:
        try:
            sequence = 0
            while not cancelled.is_set() and (config.max_frames is None or sequence < config.max_frames):
                read_start = clock()
                if not first_read_start:
                    first_read_start.append(read_start)
                    first_read_started.set()
                captured = capture.read()
                read_end = clock()
                if not isinstance(captured, tuple) or len(captured) != 2:
                    raise RuntimeError("capture.read() must return (success, frame)")
                ok, frame = captured
                if not ok or frame is None:
                    break
                packet = (sequence, frame, read_start, read_end, wall_clock())
                sequence += 1
                if not put_bounded(frames, packet):
                    return
            put_bounded(frames, eof)
        except BaseException as exc:
            fail("capture", exc)
        finally:
            try:
                capture.release()
            except BaseException as exc:
                fail("capture release", exc)

    def inference_worker() -> None:
        try:
            while not cancelled.is_set():
                try:
                    item = frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is eof:
                    put_bounded(results, done)
                    return
                frame_id, frame, read_start, read_end, capture_unix = item
                dequeued = clock()
                synchronizer(config.device)
                predict_start = clock()
                predictions = predictor(frame)
                synchronizer(config.device)
                predict_end = clock()
                packet = (frame_id, frame, predictions, read_start, read_end, capture_unix, dequeued, predict_end, (predict_end - predict_start) * 1000)
                if not put_bounded(results, packet):
                    return
        except BaseException as exc:
            fail("predictor", exc)
        finally:
            put_bounded(results, done)

    try:
        if config.warmup:
            np = importlib.import_module("numpy")
            dummy = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
            for _ in range(config.warmup):
                synchronizer(config.device)
                predictor(dummy)
                synchronizer(config.device)
        capture_thread = threading.Thread(target=capture_worker, name="vision-capture", daemon=True)
        inference_thread = threading.Thread(target=inference_worker, name="vision-inference", daemon=True)
        capture_thread.start()
        capture_owned_by_worker = True
        inference_thread.start()
        first_read_started.wait()
        started = first_read_start[0]
        inference_done = False
        while not inference_done:
            with errors_lock:
                current_error = errors[0] if errors else None
            if current_error is not None:
                stage, exc = current_error
                raise RuntimeError(f"threaded {stage} failed: {exc}") from exc
            try:
                item = results.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is done:
                inference_done = True
                continue
            frame_id, frame, predictions, read_start, read_end, capture_unix, dequeued, predict_end, predict_wall_ms = item
            annotate_started = clock()
            annotated = renderer(frame, predictions)
            annotation_finished = clock()
            display_started = annotation_finished
            if config.show:
                display_started = clock()
                assert display is not None and wait_key is not None
                display("Threaded vision pipeline", annotated)
                key = wait_key(1)
                completed = clock()
                if key & 0xFF == ord("q"):
                    stop_reason = "user_quit"
            else:
                completed = annotation_finished
            queue_wait_ms = ((dequeued - read_end) + (annotate_started - predict_end)) * 1000
            resource = resource_sampler() if resource_sampler else {"process_cpu_percent": None, "rss_bytes": None}
            rows.append({
                "frame_id": frame_id,
                "capture_time_unix_s": capture_unix,
                "capture_ms": (read_end - read_start) * 1000,
                "queue_wait_ms": queue_wait_ms,
                "preprocess_ms": _result_speed(predictions)[0],
                "inference_ms": _result_speed(predictions)[1],
                "postprocess_ms": _result_speed(predictions)[2],
                "predict_wall_ms": predict_wall_ms,
                "annotation_ms": (annotation_finished - annotate_started) * 1000,
                "display_ms": (completed - display_started) * 1000 if config.show else None,
                "annotation_display_ms": (completed - annotate_started) * 1000,
                "end_to_end_ms": (completed - read_start) * 1000,
                "completion_offset_s": completed - started,
                "detection_count": _detection_count(predictions),
                "process_cpu_percent": resource.get("process_cpu_percent"),
                "rss_bytes": resource.get("rss_bytes"),
            })
            if stop_reason == "user_quit":
                measured_end = completed
                cancelled.set()
                break
        if measured_end is None:
            measured_end = clock()
    except Exception as exc:
        cancelled.set()
        raise RuntimeError(f"threaded pipeline failed: {exc}") from exc
    finally:
        cancelled.set()
        if capture_thread is not None:
            capture_thread.join(join_timeout)
        if not capture_owned_by_worker:
            with contextlib.suppress(Exception):
                capture.release()
        if inference_thread is not None:
            inference_thread.join(join_timeout)
        if config.show:
            with contextlib.suppress(ImportError):
                importlib.import_module("cv2").destroyAllWindows()

    threads = [thread for thread in (capture_thread, inference_thread) if thread is not None]
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError(f"threaded pipeline could not stop worker(s): {', '.join(alive)}; native capture.read() may be blocked and cannot be forcibly interrupted")
    with errors_lock:
        late_error = errors[0] if errors else None
    if late_error is not None:
        stage, exc = late_error
        raise RuntimeError(f"threaded {stage} failed: {exc}") from exc
    if measured_end is None:
        measured_end = started if started is not None else clock()
    return _assemble_run(config, rows, max(measured_end - started, 0.0) if started is not None else 0.0, stop_reason, source_fps, model_name, hardware)


def _versions() -> dict[str, str | None]:
    import importlib.metadata
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("ultralytics", "opencv-python", "numpy", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware(device: str) -> dict[str, Any]:
    data: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(), "requested_device": device or "auto"}
    try:
        torch = importlib.import_module("torch")
        data["torch"] = torch.__version__
        data["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            data["accelerator"] = torch.cuda.get_device_name(0)
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            data["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        data["torch"] = None
    return data


def _resource_sampler() -> Callable[[], dict[str, float | int | None]]:
    try:
        psutil = importlib.import_module("psutil")
        process = psutil.Process(os.getpid())
        process.cpu_percent(None)
        return lambda: {"process_cpu_percent": process.cpu_percent(None), "rss_bytes": process.memory_info().rss}
    except ImportError:
        return lambda: {"process_cpu_percent": None, "rss_bytes": None}


def export_run(result: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "frames.csv"
    json_path = directory / "summary.json"
    rows = result["rows"]
    fields = list(rows[0]) if rows else [
        "frame_id", "capture_time_unix_s", "capture_ms", "queue_wait_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
        "predict_wall_ms", "annotation_ms", "display_ms", "annotation_display_ms", "end_to_end_ms", "completion_offset_s", "detection_count",
        "process_cpu_percent", "rss_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2, allow_nan=False)
        handle.write("\n")
    return csv_path, json_path


def run_real(config: PipelineConfig) -> dict[str, Any]:
    """Create a real OpenCV/Ultralytics run; setup and model warm-up precede timing."""
    _validate_config(config, config.mode)
    if config.mode not in ("sync", "threaded"):
        raise ValueError("mode must be sync or threaded")
    capture = open_capture(config.source)
    owned_by_pipeline = False
    try:
        fps = capture.get(5) or None
        predictor, renderer, model_name = load_yolo(config.weights, config.device, config.imgsz, config.conf)
        runner = run_threaded if config.mode == "threaded" else run_sync
        display = wait_key = None
        if config.show:
            cv2 = importlib.import_module("cv2")
            display = cv2.imshow
            wait_key = cv2.waitKey
        owned_by_pipeline = True
        return runner(
            config, capture, predictor, renderer, source_fps=fps,
            model_name=model_name, hardware=_hardware(config.device),
            resource_sampler=_resource_sampler(), display=display, wait_key=wait_key,
        )
    except BaseException:
        # The pipeline owns capture once entered; before entry (e.g. model load
        # failure), this adapter remains responsible for closing the handle.
        if not owned_by_pipeline:
            with contextlib.suppress(Exception):
                capture.release()
        raise


def open_capture(source: str) -> Any:
    cv2 = importlib.import_module("cv2")
    parsed: str | int = int(source) if source.isdecimal() else source
    capture = cv2.VideoCapture(parsed)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open video source {source!r}")
    return capture


def load_yolo(weights: str, device: str, imgsz: int, conf: float) -> tuple[Callable[[Any], Any], Callable[[Any, Any], Any], str]:
    """Import Ultralytics only on execution; --help and tests need no model download."""
    YOLO = importlib.import_module("ultralytics").YOLO
    model = YOLO(weights)
    def predict(frame: Any) -> Any:
        return model.predict(frame, imgsz=imgsz, device=device or None, conf=conf, verbose=False)
    def render(frame: Any, results: Any) -> Any:
        items = results if isinstance(results, (list, tuple)) else [results]
        return items[0].plot() if items else frame
    return predict, render, getattr(model, "task", "unknown")


def main(argv: Iterable[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run a synchronous or bounded-threaded YOLO video pipeline and export timing data.")
    parser.add_argument("source", help="video path or numeric webcam index (for example 0)")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--mode", choices=("sync", "threaded"), default="sync")
    parser.add_argument("--queue-size", type=int, default=2, help="positive capacity for each threaded FIFO queue")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="auto, cpu, mps, cuda, or a CUDA index")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=0, help="number of dummy-image model warm-up predictions")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--show", action="store_true", help="display annotated frames; q stops capture")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.warmup < 0 or args.queue_size <= 0 or not 0 <= args.conf <= 1 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("imgsz and queue-size must be positive, warmup non-negative, conf in [0, 1], and max-frames positive")
    config = PipelineConfig(args.source, args.weights, args.mode, args.imgsz, args.device, args.conf, args.warmup, args.max_frames, args.show, args.output_dir, args.queue_size)
    capture = None
    runner_called = False
    try:
        capture = open_capture(args.source)
        fps = capture.get(5) or None
        predictor, renderer, _task = load_yolo(args.weights, args.device, args.imgsz, args.conf)
        display = wait_key = None
        if args.show:
            cv2 = importlib.import_module("cv2")
            display = cv2.imshow
            wait_key = cv2.waitKey
        runner = run_threaded if args.mode == "threaded" else run_sync
        runner_called = True
        result = runner(config, capture, predictor, renderer, source_fps=fps, hardware=_hardware(config.device), resource_sampler=_resource_sampler(), display=display, wait_key=wait_key)
        csv_path, json_path = export_run(result, args.output_dir)
        print(f"Frames: {result['summary']['frames']} | annotated throughput: {result['summary']['annotated_throughput_fps']}")
        print(f"CSV: {csv_path}\nSummary: {json_path}")
        return 0
    except Exception as exc:
        if capture is not None and not runner_called:
            capture.release()
        detail = f"required runtime dependency unavailable ({exc}); install project dependencies from requirements.txt" if isinstance(exc, ImportError) else str(exc)
        print(f"pipeline error: {detail}", file=__import__("sys").stderr)
        return 1
