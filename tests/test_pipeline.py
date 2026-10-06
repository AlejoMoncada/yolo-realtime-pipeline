from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import vision_pipeline
from vision_pipeline import PipelineConfig, export_run, run_real, run_sync, run_threaded

ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACTS = ROOT / ".cache" / "tests"


class FakeResult:
    def __init__(self, value: int, speed=None):
        self.boxes = [object()] * value
        self.speed = speed or {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}


class FakeCapture:
    def __init__(self, frames, events=None, error=None):
        self.frames = list(frames)
        self.events = events if events is not None else []
        self.error = error
        self.released = False

    def get(self, _property):
        return 0

    def read(self):
        self.events.append("read")
        if self.error:
            raise self.error
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.events.append("release")
        self.released = True


class PipelineTests(unittest.TestCase):
    def config(self, **kwargs):
        return PipelineConfig("fake.mp4", **kwargs)

    def run_fake(self, frames, **kwargs):
        events = []
        capture = FakeCapture(frames, events)

        def predict(frame):
            events.append(("predict", frame))
            return [FakeResult(frame)]

        def render(frame, results):
            events.append(("render", frame, results[0].boxes.__len__()))
            return f"annotated-{frame}"

        clocks = iter(i / 100 for i in range(1000))
        result = run_sync(
            self.config(**kwargs), capture, predict, render,
            clock=lambda: next(clocks), wall_clock=lambda: 123.5,
            synchronizer=lambda device: None,
        )
        return result, capture, events

    def test_pairs_each_capture_with_its_own_prediction_and_render_in_order(self):
        result, capture, events = self.run_fake([10, 20])
        self.assertEqual([r["frame_id"] for r in result["rows"]], [0, 1])
        self.assertEqual([e for e in events if isinstance(e, tuple)], [
            ("predict", 10), ("render", 10, 10), ("predict", 20), ("render", 20, 20),
        ])
        self.assertEqual([r["detection_count"] for r in result["rows"]], [10, 20])
        self.assertTrue(capture.released)
        self.assertLess(events.index("release"), len(events))

    def test_max_frames_stops_without_reading_an_extra_frame(self):
        result, capture, events = self.run_fake([1, 2, 3], max_frames=2)
        self.assertEqual(result["summary"]["frames"], 2)
        self.assertEqual(events.count("read"), 2)
        self.assertTrue(capture.released)

    def test_empty_source_and_eof_produce_valid_empty_result(self):
        result, _, events = self.run_fake([])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["summary"]["stop_reason"], "eof")
        self.assertEqual(events.count("read"), 1)

    def test_predictor_error_is_readable_and_capture_is_released(self):
        capture = FakeCapture([1])
        def fail(_frame):
            raise ValueError("broken model")
        with self.assertRaisesRegex(RuntimeError, "synchronous pipeline failed: broken model"):
            run_sync(self.config(), capture, fail, lambda f, p: f, synchronizer=lambda _: None)
        self.assertTrue(capture.released)

    def test_capture_error_is_readable_and_capture_is_released(self):
        capture = FakeCapture([], error=OSError("camera disconnected"))
        with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
            run_sync(self.config(), capture, lambda f: [], lambda f, p: f)
        self.assertTrue(capture.released)

    def test_renderer_error_is_readable_and_capture_is_released(self):
        capture = FakeCapture([1])
        with self.assertRaisesRegex(RuntimeError, "draw failed"):
            run_sync(self.config(), capture, lambda f: [FakeResult(0)], lambda f, p: (_ for _ in ()).throw(ValueError("draw failed")), synchronizer=lambda _: None)
        self.assertTrue(capture.released)

    def test_warmup_uses_dummy_frames_before_source_and_does_not_consume_them(self):
        capture = FakeCapture([9])
        seen = []
        def predict(frame):
            seen.append(frame)
            return [FakeResult(0)]
        class DummyImage:
            shape = (640, 640, 3)
        fake_numpy = type("FakeNumpy", (), {"uint8": object(), "zeros": staticmethod(lambda shape, dtype: DummyImage())})
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            run_sync(self.config(warmup=2), capture, predict, lambda f, p: f, synchronizer=lambda _: None)
        self.assertEqual(seen[-1], 9)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0].shape, (640, 640, 3))
        self.assertEqual(seen[1].shape, (640, 640, 3))

    def threaded_config(self, **kwargs):
        return PipelineConfig("fake.mp4", mode="threaded", queue_size=1, **kwargs)

    def test_threaded_queue_one_slow_stages_preserve_pairing_order_and_ownership(self):
        import threading
        import time

        capture = FakeCapture(range(12))
        capture_thread_ids = []
        original_read = capture.read
        original_release = capture.release
        capture.read = lambda: (capture_thread_ids.append(threading.get_ident()), original_read())[1]
        capture.release = lambda: (capture_thread_ids.append(threading.get_ident()), original_release())[1]
        main_thread = threading.get_ident()
        inference_threads = []
        render_threads = []

        def predict(frame):
            inference_threads.append(threading.get_ident())
            time.sleep(0.002)
            return [FakeResult(frame)]

        def render(frame, prediction):
            render_threads.append(threading.get_ident())
            time.sleep(0.002)
            return (frame, prediction[0].boxes.__len__())

        result = run_threaded(self.threaded_config(), capture, predict, render, synchronizer=lambda _: None)
        self.assertEqual([row["frame_id"] for row in result["rows"]], list(range(12)))
        self.assertEqual([row["detection_count"] for row in result["rows"]], list(range(12)))
        self.assertEqual(capture.events.count("read"), 13)  # includes one EOF read
        self.assertEqual(len(capture_thread_ids), 14)  # reads plus owner-thread release
        self.assertEqual(len(set(capture_thread_ids)), 1)
        self.assertNotEqual(capture_thread_ids[0], main_thread)
        self.assertEqual(len(set(inference_threads)), 1)
        self.assertNotEqual(inference_threads[0], main_thread)
        self.assertEqual(set(render_threads), {main_thread})
        self.assertTrue(capture.released)
        self.assertFalse(any(thread.name in {"vision-capture", "vision-inference"} for thread in threading.enumerate()))
        self.assertEqual(result["summary"]["stop_reason"], "eof")

    def test_threaded_max_frames_reads_exact_limit_and_joins_workers(self):
        capture = FakeCapture([4, 5, 6, 7])
        result = run_threaded(self.threaded_config(max_frames=2), capture, lambda f: [FakeResult(f)], lambda f, p: f, synchronizer=lambda _: None)
        self.assertEqual([row["frame_id"] for row in result["rows"]], [0, 1])
        self.assertEqual(capture.events.count("read"), 2)
        self.assertEqual(capture.events[-1], "release")

    def test_threaded_warmup_failures_release_before_capture_ownership_transfers(self):
        import threading

        class CountingCapture(FakeCapture):
            def __init__(self):
                super().__init__([1])
                self.release_threads = []
            def release(self):
                self.release_threads.append(threading.get_ident())
                super().release()

        main_thread = threading.get_ident()
        fake_numpy = type("FakeNumpy", (), {"uint8": object(), "zeros": staticmethod(lambda shape, dtype: object())})
        for predictor, expected in (
            (lambda _frame: (_ for _ in ()).throw(ValueError("warmup failed")), "warmup failed"),
        ):
            capture = CountingCapture()
            with patch.dict(sys.modules, {"numpy": fake_numpy}), self.assertRaisesRegex(RuntimeError, expected):
                run_threaded(self.threaded_config(warmup=1), capture, predictor, lambda f, p: f)
            self.assertEqual(len(capture.release_threads), 1)
            self.assertEqual(capture.release_threads[0], main_thread)

        capture = CountingCapture()
        broken_numpy = type("BrokenNumpy", (), {"uint8": object(), "zeros": staticmethod(lambda shape, dtype: (_ for _ in ()).throw(ValueError("dummy failed")))})
        with patch.dict(sys.modules, {"numpy": broken_numpy}), self.assertRaisesRegex(RuntimeError, "dummy failed"):
            run_threaded(self.threaded_config(warmup=1), capture, lambda f: [], lambda f, p: f)
        self.assertEqual(capture.release_threads, [main_thread])

    def test_threaded_invalid_real_config_does_not_open_capture(self):
        with patch.object(vision_pipeline, "open_capture") as open_source, self.assertRaisesRegex(ValueError, "imgsz.*positive"):
            run_real(self.threaded_config(imgsz=0))
        open_source.assert_not_called()

    def test_run_real_forwards_display_callbacks_only_for_showing_in_both_modes(self):
        for mode, runner_name in (("sync", "run_sync"), ("threaded", "run_threaded")):
            for show in (False, True):
                with self.subTest(mode=mode, show=show):
                    config = PipelineConfig("fake.mp4", mode=mode, show=show)
                    capture = FakeCapture([])
                    def display(title, frame):
                        return None
                    def wait_key(delay):
                        return -1
                    fake_cv2 = type("FakeCV2", (), {"imshow": staticmethod(display), "waitKey": staticmethod(wait_key)})
                    with (
                        patch.object(vision_pipeline, "open_capture", return_value=capture),
                        patch.object(vision_pipeline, "load_yolo", return_value=(lambda f: [], lambda f, p: f, "detect")),
                        patch.object(vision_pipeline, runner_name, return_value={"ok": True}) as runner,
                        patch.dict(sys.modules, {"cv2": fake_cv2}),
                    ):
                        self.assertEqual(run_real(config), {"ok": True})
                    kwargs = runner.call_args.kwargs
                    self.assertEqual(kwargs.get("display"), display if show else None)
                    self.assertEqual(kwargs.get("wait_key"), wait_key if show else None)

    def test_cli_dependency_error_is_concise_and_releases_capture(self):
        import io
        capture = FakeCapture([])
        stderr = io.StringIO()
        with (
            patch.object(vision_pipeline, "open_capture", return_value=capture),
            patch.object(vision_pipeline, "load_yolo", side_effect=ModuleNotFoundError("No module named 'ultralytics'")),
            patch("sys.stderr", stderr),
        ):
            code = vision_pipeline.main(["fake.mp4"])
        self.assertEqual(code, 1)
        self.assertTrue(capture.released)
        self.assertIn("install project dependencies", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_threaded_capture_predictor_and_renderer_failures_propagate_and_release(self):
        capture = FakeCapture([], error=OSError("camera disconnected"))
        with self.assertRaisesRegex(RuntimeError, "threaded capture failed: camera disconnected"):
            run_threaded(self.threaded_config(), capture, lambda f: [], lambda f, p: f)
        self.assertTrue(capture.released)

        capture = FakeCapture([1, 2, 3])
        def broken_predictor(_frame):
            raise ValueError("broken model")
        with self.assertRaisesRegex(RuntimeError, "threaded predictor failed: broken model"):
            run_threaded(self.threaded_config(), capture, broken_predictor, lambda f, p: f, synchronizer=lambda _: None)
        self.assertTrue(capture.released)

        capture = FakeCapture([1])
        def broken_renderer(_frame, _prediction):
            raise ValueError("draw failed")
        with self.assertRaisesRegex(RuntimeError, "threaded pipeline failed: draw failed"):
            run_threaded(self.threaded_config(), capture, lambda f: [FakeResult(1)], broken_renderer, synchronizer=lambda _: None)
        self.assertTrue(capture.released)

    def test_threaded_q_cancels_with_full_capacity_and_preserves_release_ownership(self):
        import threading
        import time

        capture = FakeCapture(range(100))
        release_threads = []
        original_release = capture.release
        capture.release = lambda: (release_threads.append(threading.get_ident()), original_release())[1]
        main_thread = threading.get_ident()
        result = run_threaded(
            self.threaded_config(show=True), capture, lambda f: [FakeResult(f)], lambda f, p: f,
            display=lambda title, frame: time.sleep(0.002), wait_key=lambda delay: ord("q"),
            synchronizer=lambda _: None,
        )
        self.assertEqual(result["summary"]["stop_reason"], "user_quit")
        self.assertGreaterEqual(len(result["rows"]), 1)
        self.assertLess(len(result["rows"]), 100)
        self.assertEqual(len(release_threads), 1)
        self.assertNotEqual(release_threads[0], main_thread)
        self.assertTrue(capture.released)

    def test_threaded_blocked_capture_read_is_reported_not_claimed_stopped(self):
        import threading

        unblock = threading.Event()
        released = threading.Event()
        class BlockingCapture:
            def __init__(self):
                self.reads = 0
            def read(self):
                self.reads += 1
                if self.reads == 1:
                    return True, 1
                unblock.wait()
                return False, None
            def release(self):
                released.set()

        capture = BlockingCapture()
        with self.assertRaisesRegex(RuntimeError, r"could not stop worker.*capture\.read\(\) may be blocked"):
            run_threaded(
                self.threaded_config(show=True), capture, lambda f: [], lambda f, p: f,
                wait_key=lambda delay: ord("q"), display=lambda title, frame: None,
                join_timeout=0.02,
            )
        unblock.set()
        self.assertTrue(released.wait(1))

    def test_threaded_stage_fields_and_queue_wait_are_finite(self):
        import time

        capture = FakeCapture([1, 2, 3])
        def predict(frame):
            time.sleep(0.004)
            return [FakeResult(frame)]
        result = run_threaded(self.threaded_config(), capture, predict, lambda f, p: f, synchronizer=lambda _: None)
        for row in result["rows"]:
            for name in ("capture_ms", "queue_wait_ms", "predict_wall_ms", "annotation_ms", "end_to_end_ms"):
                self.assertTrue(math.isfinite(row[name]), (name, row[name]))
            self.assertIsNone(row["display_ms"])
            self.assertGreaterEqual(row["queue_wait_ms"], 0)
        self.assertTrue(any(row["queue_wait_ms"] > 0 for row in result["rows"]))
        stages = result["summary"]["stage_timings_ms"]
        self.assertIsNotNone(stages["inference_ms"]["mean"])
        self.assertIsNone(stages["display_ms"]["mean"])
        self.assertIn("worker shutdown", result["summary"]["timing_duration"])

    def test_queue_wait_is_zero_and_predict_wall_is_not_internal_inference(self):
        result, _, _ = self.run_fake([1])
        row = result["rows"][0]
        self.assertEqual(row["queue_wait_ms"], 0.0)
        self.assertEqual(row["inference_ms"], 2.0)
        self.assertNotEqual(row["predict_wall_ms"], row["inference_ms"])
        self.assertEqual(row["capture_time_unix_s"], 123.5)
        self.assertTrue(math.isfinite(row["capture_ms"]))
        self.assertTrue(math.isfinite(row["annotation_ms"]))
        self.assertIsNone(row["display_ms"])
        self.assertIn("inference_ms", result["summary"]["stage_timings_ms"])

    def test_display_fps_is_null_headless_and_configured_when_displayed(self):
        result, _, _ = self.run_fake([1])
        self.assertIsNone(result["summary"]["display_fps"])
        capture = FakeCapture([1])
        shown = []
        result = run_sync(
            self.config(show=True), capture, lambda f: [FakeResult(1)], lambda f, p: "marked",
            display=lambda title, frame: shown.append(frame), wait_key=lambda delay: -1,
            synchronizer=lambda _: None,
        )
        self.assertEqual(shown, ["marked"])
        self.assertIsNotNone(result["summary"]["display_fps"])

    def test_invalid_config_and_pending_mode_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "only --mode sync"):
            run_sync(self.config(mode="threaded"), FakeCapture([]), lambda f: [], lambda f, p: f)
        with self.assertRaisesRegex(ValueError, "imgsz must be positive"):
            run_sync(self.config(imgsz=0), FakeCapture([]), lambda f: [], lambda f, p: f)

    def test_export_writes_csv_and_json_and_handles_zero_frames(self):
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=TEST_ARTIFACTS) as directory:
            result, _, _ = self.run_fake([])
            csv_path, json_path = export_run(result, directory)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 0)
            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["frames"], 0)
            self.assertEqual(summary["complete_stability_windows"], [])

    def test_summary_statistics_and_complete_windows_include_empty_seconds(self):
        capture = FakeCapture([1, 2])
        ticks = iter([0, .1, .1, .1, .1, .4, .5, .9, .9, .9, .9, 2.4, 3.1])
        result = run_sync(
            self.config(max_frames=2), capture, lambda f: [FakeResult(1)], lambda f, p: f,
            clock=lambda: next(ticks), wall_clock=lambda: 0, synchronizer=lambda _: None,
        )
        summary = result["summary"]
        self.assertEqual(summary["latency_end_to_end_ms"]["p50"], 1150.0)
        self.assertEqual(summary["complete_stability_windows"], [1, 0, 1])
        self.assertIs(summary["stability_windows_exclude_trailing_partial"], True)

    def test_cli_help_does_not_import_model_runtime(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, str(ROOT / "pipeline.py"), "--help"], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--max-frames", completed.stdout)
        self.assertNotIn("ultralytics", completed.stderr.lower())

    def test_cli_rejects_invalid_numeric_options(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "pipeline.py"), "clip.mp4", "--conf", "1.2"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("conf in [0, 1]", completed.stderr)


if __name__ == "__main__":
    unittest.main()
