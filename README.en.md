# Real-time vision pipeline

A YOLO pipeline with a synchronous reference mode and a bounded, lossless threaded mode. `benchmark.py` compares both modes on the same local video and frame IDs. Results are measurements, not promises of improvement or a target frame rate.

See also the [Spanish README](README.md) and the [detailed English summary](docs/implementation-summary.en.md). The [Spanish summary](docs/implementation-summary.es.md) contains the same technical facts.

## Install, run, benchmark, and test

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline.py ./clip.mp4 --mode threaded --imgsz 640 --device auto --conf 0.25 --warmup 2 --max-frames 300 --queue-size 2 --output-dir results
.venv/bin/python benchmark.py ./clip.mp4 --weights yolo11n.pt --device auto --conf 0.25 --imgsz 640 --warmup 2 --max-frames 300 --queue-size 2 --repeats 3 --output-dir benchmark-results
.venv/bin/python -m unittest discover -s tests -v
```

To use pip instead of uv, create a Python 3.12 virtual environment with `python3.12 -m venv .venv`. Then install with `.venv/bin/python -m pip install -r requirements.txt`.

Weights such as `yolo11n.pt` may download during the first real inference. `--help` does not import Ultralytics or load weights. Review Ultralytics' [licensing options](https://www.ultralytics.com/license) and [official documentation](https://docs.ultralytics.com/) for license terms and attribution obligations that apply to your use.

## Controlled benchmark

The benchmark accepts only an existing local video file. It does not accept a webcam index or network source. For example:

```sh
.venv/bin/python benchmark.py ./clip.mp4 --repeats 5 --output-dir benchmark-results
.venv/bin/python benchmark.py ./clip.mp4 --imgsz 640 --optimized-imgsz 480 --repeats 3
```

### ONNX format and camera pacing

`export_model.py` exports weights at a fixed input size and names the file by resolution, for example `yolo11n-320.onnx`. `--optimized-weights` applies those weights to the optimized case only and requires `--optimized-imgsz` at the exported resolution:

```sh
.venv/bin/python export_model.py yolo11n.pt --imgsz 320
.venv/bin/python benchmark.py ./clip.mp4 --device cpu --imgsz 640 --optimized-imgsz 320 --optimized-weights yolo11n-320.onnx --repeats 3
```

Without `--realtime`, the benchmark reads the file as fast as possible: queues stay full and `end_to_end_ms` includes that backpressure. With `--realtime` (in `benchmark.py` and `pipeline.py`), frame `n` is released at `origin + n / source FPS`, as if it arrived from a camera. `end_to_end_ms` is then measured from that scheduled arrival and includes any accumulated backlog, and `capture_lag_ms` reports how late each frame was read relative to its arrival. Throughput is capped by the source FPS, so this mode measures latency, not maximum throughput. `--realtime` requires the source to report a positive FPS.

Each invocation writes to a unique `benchmark-<timestamp>-<id>/` child directory. It hashes the source before timing and creates a subdirectory for each run. Each run contains `frames.csv` and `summary.json`. `comparison.json` records:

- Input identity: resolved path, byte size, and SHA-256; selected frame count and IDs.
- Parameters, environment metadata, and exact repeat order.
- For each case: throughput per run, median throughput, per-run E2E p95 and its median, speedup, percentage change versus sync, and relative artifact paths.

Throughput is unique completed frames divided by the run's measured elapsed time. The reported value is the median across repeats. Each run's E2E p95 spans from source-read start through annotation and display. It includes queue wait. The comparison reports the median of those p95 values. Speedup is case median throughput divided by sync median throughput. Percentage change is `(case / sync - 1) * 100`. Model initialization and dummy-image warm-up are outside each measurement. Case order rotates across repeats, so a case does not always use the same cache position.

Sync and threaded cases use the same weights, device, confidence, inference resolution, and frame IDs. `--optimized-imgsz` adds a separate threaded case at a lower resolution. Its throughput does not measure concurrency alone or prove predictive-quality parity. No accuracy or quality claim is made without reference data. The benchmark rejects empty, partial, or mismatched frame results. It also rejects sources that do not report a positive frame count.

## Pipeline behavior and timing

- Sync performs read, predict, annotate, and optional display in sequence. In threaded mode, one thread captures, reads, and releases the source; one thread calls prediction; the main thread annotates and displays results. Two bounded FIFO queues preserve order and do not drop frames.
- A larger queue may improve overlap when stages have different workloads. It may also increase queue latency and memory use. FIFO backpressure preserves order and avoids drops, but does not guarantee higher throughput.
- End of video and `--max-frames` stop normally. `q` cancels pending work after the result being displayed. Worker errors propagate. A blocked native `capture.read()` call cannot be forcibly interrupted. The program reports if it remains blocked and does not claim it stopped.
- `capture_ms` measures source reading. `preprocess_ms`, `inference_ms`, and `postprocess_ms` are Ultralytics `Results.speed` values in milliseconds per image; they may be `null`. `predict_wall_ms` measures the full synchronized predict call, not inference alone.
- `queue_wait_ms` includes waiting from capture to inference and from result to annotation, including backpressure. `annotation_ms` and `display_ms` are separate measurements. `display_ms` is `null` in headless mode. `end_to_end_ms` covers reading, queue waits, prediction, annotation, and optional display.
- Total elapsed time starts at the first measured read. It excludes model setup, warm-up, worker shutdown, and cleanup. CSV rows contain IDs, stage timings, detection count, and process CPU/RSS when available. Summaries include mean, p50, p95, and p99, plus complete one-second stability windows. Offline throughput is not source or playback FPS. Headless annotated throughput is not the monitor's physical refresh rate.

## Synthetic wiring clip

`examples/make_demo_video.py [output]` creates a short OpenCV clip with labeled moving shapes. By default, it has 48 frames at 320×240 and 24 FPS. Frame count, dimensions, and FPS are configurable. `--help` works without OpenCV. Generation requires installed OpenCV and NumPy. This synthetic clip only checks capture-to-annotation wiring. It does not prove accuracy or realistic YOLO performance. No quality claim is based on synthetic data.

## Local validation status

Activity 4 report (Spanish): [docs/Informe_Actividad4_Grupo9_Optimizacion_TiempoReal.pdf](docs/Informe_Actividad4_Grupo9_Optimizacion_TiempoReal.pdf).

Tests: `.venv/bin/python -m unittest discover -s tests` runs 47 fake-backend tests. Real measurements (macOS arm64, CPU, Python 3.12, Ultralytics 8.4, ONNX Runtime 1.31, `yolo11n`) use the same `car-detection.mp4` video (768×432, 12.5 FPS, 377 frames, SHA-256 `d31e0ebf…cca34`) and 3 repeats per case:

| Run | Case | Median throughput | Median E2E p95 |
|---|---|---|---|
| Offline `benchmark-20261012-171741-64736e26` | sync, `.pt`, 640 | 49.1 FPS | 27.3 ms |
| | threaded, `.pt`, 640 | 42.4 FPS | 126.9 ms |
| | threaded, ONNX, 320 | 138.3 FPS | 37.2 ms |
| `--realtime` `benchmark-20261012-172526-e682279c` | sync, `.pt`, 640 | 12.5 FPS | 34.4 ms |
| | threaded, `.pt`, 640 | 12.5 FPS | 45.3 ms |
| | threaded, ONNX, 320 | 12.5 FPS | 13.6 ms |

The jump above 60 FPS comes from the lower resolution and ONNX, not from concurrency alone: with inference dominating each frame, threads at 640 do not raise offline throughput and their full queue raises latency. At source pace, queues stay nearly empty and latency depends mostly on inference. Throughput varies between runs. Headless, `display_fps` is `null`: these figures are annotated throughput, not physical display FPS. `detection_count` does not replace a labeled quality evaluation.
