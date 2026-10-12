# Implementation summary: real-time vision pipeline

This document explains the implementation, available measurements, and their limits. It is for students who need to run the project and understand what each result proves. See also the [English README](../README.en.md) and the [Spanish summary](implementation-summary.es.md).

This text follows clarity principles inspired by ASD-STE100: direct sentences, defined terms, and one main idea per sentence. The requested 80% simplification target is approximate. It is not a measured score or certification. ASD-STE100 focuses on English, so the Spanish version adapts its clarity principles. The text was not checked against an approved dictionary or formally audited against a standard version. The official site is background, not proof of compliance: [ASD-STE100](https://www.asd-ste100.org/).

## 1. What was built

The project started without an implementation prototype. `pipeline.py` is the command-line entry point. `vision_pipeline.py` contains `PipelineConfig`, the `run_sync`, `run_threaded`, and `run_real` functions, metric helpers, and `export_run`. [`benchmark.py`](../benchmark.py) controls comparisons. Behavioral tests are in [`tests/`](../tests/). [`examples/make_demo_video.py`](../examples/make_demo_video.py) creates a wiring video with synthetic shapes. [`requirements.txt`](../requirements.txt) declares four dependency groups: Ultralytics, OpenCV, NumPy, and psutil.

Sync is a simple reference mode. Threaded mode tries to overlap reading and prediction. Both modes produce annotations and per-frame measurements. The design does not promise a universal improvement. Performance depends on the video, model, device, and cost of each stage.

## 2. How each mode works

In sync mode, one flow reads a frame, calls the predictor, annotates the result, and displays it when requested. These stages do not overlap.

In threaded mode, the normal flow is:

1. The capture thread reads frames and assigns increasing IDs. It alone uses and releases the capture after ownership transfer.
2. A bounded FIFO queue sends each ID and image pair to the inference worker.
3. One worker calls the predictor. The same model is not called concurrently.
4. A second bounded FIFO queue sends the ID, original image, and result to the main thread.
5. The main thread annotates the matching image and displays it when requested.

The queues are bounded and do not drop frames. They preserve order and image-to-prediction pairing. If one stage runs faster than the next, backpressure limits accumulation. A larger queue may allow more overlap, but can also increase latency.

Queue size limits frames waiting in queues, not all process memory. Measurement rows accumulate in memory until export. Total memory use can therefore grow with run duration. No speedup from the GIL or specific hardware was measured or promised.

Warm-up uses dummy images before the first measured read. In threaded mode, it happens before the threads start. Therefore, not every model call always occurs inside the inference worker. Setup, warm-up, shutdown, and cleanup are outside measured time.

`--max-frames` limits the number of processed frames. End of video stops normally. The `q` key requests cancellation during display. Worker errors reach the main thread. A blocked native `capture.read()` call cannot be safely interrupted. Before threads start, the main thread can release a capture if setup fails. After capture starts, the main thread does not release it concurrently. If the native read does not return, the program reports the blocked worker.

## 3. What the measurements mean

`capture_ms` measures source reading time. `preprocess_ms`, `inference_ms`, and `postprocess_ms` come from Ultralytics `Results.speed` and represent milliseconds per image. These values may be null. `predict_wall_ms` measures the complete synchronized predictor call. It is not pure inference time.

`queue_wait_ms` adds waiting from capture to inference and from prediction to annotation. It includes backpressure. `annotation_ms` and `display_ms` are separate measurements. Without a display, `display_ms` is `null`. `end_to_end_ms` starts at read start and ends when annotation or display completes. It includes queue waiting. It does not measure camera exposure or upstream driver buffers.

Latency statistics report mean, p50, p95, and p99. Percentiles use linear interpolation. A `null` value means the measurement is unavailable, not zero. Throughput is unique completed frames divided by measured elapsed time. Source FPS is metadata and may differ. `display_fps` is calculated only with `--show`. It describes the software loop, not the monitor's physical refresh rate.

Stability windows cover complete one-second intervals. They include intervals with zero frames. The final incomplete interval is excluded. When psutil is available, the program samples process CPU and RSS. The first CPU sample may be zero. CPU use may exceed 100% across multiple cores. These measurements do not report GPU use. Later CSV and report writes are outside measured run time.

## 4. How the benchmark works

The benchmark accepts an existing local video file with a positive frame count reported by OpenCV. It rejects webcams, network sources, empty or partial results, and mismatched IDs. Sync and threaded use the same weights, device, confidence, resolution, and frame range. Case order rotates across repeats. Before timing, the benchmark records the file path, size, and SHA-256. It does not verify every frame's content during a normal comparison. The operator must not change the video while runs are in progress.

Each invocation creates a unique output directory. Each run contains `frames.csv` and `summary.json`. `comparison.json` records identity, configuration, run order, median throughput, median E2E p95, speedup, and percentage change. The benchmark reports observed results. It does not invent values or claim an improvement before measurement.

`--optimized-imgsz` adds a threaded case at another resolution. This case changes more than concurrency and does not prove equal quality. `--optimized-weights` lets that case use another format, for example an ONNX file exported with `export_model.py` at the same resolution. No training, quantization, or batching is implemented. `--realtime` releases each frame at the source FPS to measure latency as with a camera. The `pipeline.py` output directory can overwrite `frames.csv` and `summary.json`; use separate directories for independent runs.

## 5. Basic use and evidence

The commands below show installation, execution, a controlled comparison, and tests.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python pipeline.py ./clip.mp4 --mode threaded --imgsz 640 --device auto --conf 0.25 --warmup 2 --max-frames 300 --queue-size 2 --output-dir results
.venv/bin/python benchmark.py ./clip.mp4 --weights yolo11n.pt --device auto --conf 0.25 --imgsz 640 --warmup 2 --max-frames 300 --queue-size 2 --repeats 3 --output-dir benchmark-results
.venv/bin/python -m unittest discover -s tests -v
```

The recorded independent verification ran 32 tests with fake backends. It also used OpenCV 4.11.0 to create a synthetic 12-frame AVI and ran four cases with a deliberately fake checksum predictor and `queue-size=1`. It checked hashes and IDs, annotation-to-image pairing, capture release, and artifact readback. This checks software wiring. It does not check real YOLO, GPU, GUI, camera, accuracy, or long-duration stability.

Recorded corrections separated `capture_ms` from display and annotation timings, fixed `--show` callback forwarding, released captures when warm-up failed before thread startup, and improved missing-dependency errors. Tests with doubles do not replace a real run.

## 6. Status and next steps

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

Pending: a `--show` run on a physical display to measure `display_fps`, labeled quality validation, and long-duration stability. There is no native review approval or receipt.

## Glossary

- **FIFO:** a queue that returns the first item it received.
- **E2E:** time from read start (or, with `--realtime`, the frame's scheduled arrival) to completed annotation or display.
- **RSS:** process resident memory.
- **Speedup:** a case's median throughput divided by sync median throughput.
- **`null`:** unavailable data; it does not mean zero.

## Implementation references

- [CLI entry point and `main`](../pipeline.py)
- [`PipelineConfig`, `run_sync`, `run_threaded`, `run_real`, and `export_run`](../vision_pipeline.py)
- [`run_benchmark` and report generation](../benchmark.py)
- [Pipeline tests](../tests/test_pipeline.py) and [benchmark tests](../tests/test_benchmark.py)
- [Synthetic video generator](../examples/make_demo_video.py)
- [Ultralytics official documentation](https://docs.ultralytics.com/) and [licensing](https://www.ultralytics.com/license)