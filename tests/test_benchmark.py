from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from vision_pipeline import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]

benchmark_spec = importlib.util.spec_from_file_location("local_benchmark", ROOT / "benchmark.py")
assert benchmark_spec is not None and benchmark_spec.loader is not None
benchmark_module = importlib.util.module_from_spec(benchmark_spec)
benchmark_spec.loader.exec_module(benchmark_module)
run_benchmark = benchmark_module.run_benchmark
TEST_ARTIFACTS = ROOT / ".cache" / "tests"


def fake_result(config: PipelineConfig, *, frames: int = 4, fps: float = 10.0, latency: float = 20.0):
    rows = [{"frame_id": index, "end_to_end_ms": latency, "completion_offset_s": index / fps} for index in range(frames)]
    return {"rows": rows, "summary": {
        "frames": frames, "stop_reason": "eof", "annotated_throughput_fps": fps,
        "latency_end_to_end_ms": {"p95": latency}, "hardware": {"machine": "test"},
        "config": {"source": config.source, "weights": config.weights, "device": config.device,
                    "conf": config.conf, "imgsz": config.imgsz, "warmup": config.warmup,
                    "max_frames": config.max_frames, "queue_size": config.queue_size, "mode": config.mode},
    }}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=TEST_ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "local clip.mp4"
        self.source.write_bytes(b"inert test-video identity bytes")
        self.output = Path(self.temp.name) / "out"
        self.calls = []

    def runner(self, config):
        self.calls.append(config)
        rate = 8.0 if config.mode == "sync" else 12.0
        if config.imgsz == 320:
            rate = 16.0
        return fake_result(config, fps=rate, latency=1000 / rate)

    def run_case(self, **kwargs):
        return run_benchmark(self.source, runner=self.runner, frame_count_reader=lambda _path: 4,
                             output_dir=self.output, repeats=kwargs.pop("repeats", 2), **kwargs)

    def test_same_local_source_and_common_parameters_with_alternating_run_order(self):
        report = self.run_case(weights="test.pt", device="cpu", conf=0.4, imgsz=640, warmup=1, queue_size=3)
        self.assertEqual([(config.mode, config.imgsz) for config in self.calls], [
            ("sync", 640), ("threaded", 640), ("threaded", 640), ("sync", 640),
        ])
        self.assertEqual({config.source for config in self.calls}, {str(self.source.resolve())})
        for config in self.calls:
            self.assertEqual((config.weights, config.device, config.conf, config.warmup, config.max_frames, config.queue_size),
                             ("test.pt", "cpu", 0.4, 1, None, 3))
        self.assertEqual([entry["case"] for entry in report["run_order"]], ["sync", "threaded", "threaded", "sync"])
        self.assertEqual(report["selected_frame_ids"], [0, 1, 2, 3])
        self.assertTrue(report["input"]["sha256"])

    def test_optional_lower_resolution_is_isolated_and_disclaimed(self):
        report = self.run_case(imgsz=640, optimized_imgsz=320, repeats=1)
        self.assertEqual([(call.mode, call.imgsz) for call in self.calls], [
            ("sync", 640), ("threaded", 640), ("threaded", 320),
        ])
        self.assertIn("threaded_optimized_imgsz", report["cases"])
        self.assertEqual(report["cases"]["threaded_optimized_imgsz"]["imgsz"], 320)
        self.assertIn("does not establish predictive-quality parity", report["resolution_notice"])

    def test_realtime_flag_reaches_every_case(self):
        self.run_case(realtime=True, optimized_imgsz=320, repeats=1)
        self.assertEqual([call.realtime for call in self.calls], [True, True, True])
        self.calls.clear()
        report = self.run_case(repeats=1)
        self.assertEqual([call.realtime for call in self.calls], [False, False])
        self.assertIs(report["parameters"]["realtime"], False)

    def test_optimized_weights_apply_only_to_the_optimized_case(self):
        report = self.run_case(weights="yolo11n.pt", imgsz=640, optimized_imgsz=320, optimized_weights="yolo11n-320.onnx", repeats=1)
        self.assertEqual([(call.mode, call.imgsz, call.weights) for call in self.calls], [
            ("sync", 640, "yolo11n.pt"), ("threaded", 640, "yolo11n.pt"), ("threaded", 320, "yolo11n-320.onnx"),
        ])
        self.assertEqual(report["cases"]["threaded_optimized_imgsz"]["weights"], "yolo11n-320.onnx")
        self.assertEqual(report["cases"]["sync"]["weights"], "yolo11n.pt")
        self.assertEqual(report["parameters"]["optimized_weights"], "yolo11n-320.onnx")

    def test_optimized_weights_require_optimized_imgsz(self):
        with self.assertRaisesRegex(ValueError, "optimized-weights requires optimized-imgsz"):
            self.run_case(optimized_weights="yolo11n-320.onnx", repeats=1)
        self.assertEqual(self.calls, [])

    def test_help_lists_realtime_and_optimized_weights(self):
        completed = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--help"], cwd=ROOT,
                                   text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--realtime", completed.stdout)
        self.assertIn("--optimized-weights", completed.stdout)

    def test_summary_math_artifact_schema_and_unique_output_directories(self):
        report = self.run_case(repeats=2)
        threaded = report["cases"]["threaded"]
        self.assertEqual(threaded["median_throughput_fps"], 12.0)
        self.assertEqual(threaded["median_end_to_end_p95_ms"], 1000 / 12)
        self.assertEqual(threaded["speedup_vs_sync"], 1.5)
        self.assertEqual(threaded["throughput_change_percent_vs_sync"], 50.0)
        root = Path(report["output_directory"])
        summary = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], 1)
        for case in summary["cases"].values():
            for artifact in case["artifacts"]:
                self.assertTrue((root / artifact["csv"]).is_file())
                run_summary = json.loads((root / artifact["summary_json"]).read_text(encoding="utf-8"))
                self.assertIn("benchmark", run_summary)
        second = self.run_case(repeats=1)
        self.assertNotEqual(report["output_directory"], second["output_directory"])

    def test_rejects_empty_partial_and_mismatched_frame_results(self):
        for mutate, message in (
            (lambda result: result.update(rows=[], summary={"frames": 0, "stop_reason": "eof"}), "no frames"),
            (lambda result: result["rows"].pop(), "partial or non-contiguous"),
            (lambda result: result["rows"][1].update(frame_id=9), "partial or non-contiguous"),
        ):
            calls = 0
            def broken_runner(config, mutate=mutate):
                nonlocal calls
                calls += 1
                result = fake_result(config)
                if calls > 1:
                    mutate(result)
                return result
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                run_benchmark(self.source, runner=broken_runner, frame_count_reader=lambda _path: 4,
                              output_dir=self.output, repeats=1)

    def test_rejects_invalid_sources_and_options_before_runner(self):
        invalid_options: list[dict[str, Any]] = [{"repeats": 0}, {"imgsz": 0}, {"warmup": -1}, {"queue_size": 0}, {"max_frames": 0}, {"conf": math.nan}, {"optimized_imgsz": 0}, {"weights": "   "}]
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_benchmark(self.source, runner=self.runner, frame_count_reader=lambda _path: 4,
                              output_dir=self.output, **kwargs)
        with self.assertRaises(ValueError):
            run_benchmark(self.source.parent / "missing.mp4", runner=self.runner,
                          frame_count_reader=lambda _path: 4, output_dir=self.output)
        self.assertEqual(self.calls, [])

    def test_cli_rejects_camera_and_bad_repeat_without_loading_yolo(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for args in (("0",), (str(self.source), "--repeats", "0")):
            completed = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), *args], cwd=ROOT,
                                       env=env, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("ultralytics", completed.stderr.lower())

    def test_cli_dependency_error_is_concise_and_actionable(self):
        import io
        stderr = io.StringIO()
        with (
            patch.object(benchmark_module, "run_benchmark", side_effect=ModuleNotFoundError("No module named 'cv2'")),
            patch("sys.stderr", stderr),
        ):
            code = benchmark_module.main([str(self.source)])
        self.assertEqual(code, 1)
        self.assertIn("install project dependencies", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_help_does_not_import_model_dependencies(self):
        completed = subprocess.run([sys.executable, str(ROOT / "benchmark.py"), "--help"], cwd=ROOT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True,
                                   capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--optimized-imgsz", completed.stdout)
        self.assertNotIn("ultralytics", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
