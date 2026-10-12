from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACTS = ROOT / ".cache" / "tests"

export_spec = importlib.util.spec_from_file_location("local_export_model", ROOT / "export_model.py")
assert export_spec is not None and export_spec.loader is not None
export_module = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(export_module)


class ExportModelTests(unittest.TestCase):
    def setUp(self):
        TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=TEST_ARTIFACTS)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.exports = []

    def fake_ultralytics(self):
        test = self

        class FakeYOLO:
            def __init__(self, weights):
                self.weights = weights

            def export(self, **kwargs):
                test.exports.append((self.weights, kwargs))
                produced = test.directory / f"{Path(self.weights).stem}.{kwargs['format']}"
                produced.write_bytes(b"model")
                return str(produced)

        return type("FakeUltralytics", (), {"YOLO": FakeYOLO})

    def test_export_names_the_model_by_resolution_and_fixes_its_input_size(self):
        with patch.dict(sys.modules, {"ultralytics": self.fake_ultralytics()}):
            first = export_module.export_model(str(self.directory / "yolo11n.pt"), 320, "onnx", self.directory)
            second = export_module.export_model(str(self.directory / "yolo11n.pt"), 640, "onnx", self.directory)
        self.assertEqual((first.name, second.name), ("yolo11n-320.onnx", "yolo11n-640.onnx"))
        self.assertTrue(first.is_file() and second.is_file())
        self.assertFalse((self.directory / "yolo11n.onnx").exists())
        self.assertEqual([kwargs["imgsz"] for _, kwargs in self.exports], [320, 640])
        self.assertTrue(all(kwargs["dynamic"] is False for _, kwargs in self.exports))

    def test_cli_prints_the_exported_path(self):
        stdout = io.StringIO()
        with patch.dict(sys.modules, {"ultralytics": self.fake_ultralytics()}), patch("sys.stdout", stdout):
            code = export_module.main([str(self.directory / "yolo11n.pt"), "--imgsz", "320", "--output-dir", str(self.directory)])
        self.assertEqual(code, 0)
        self.assertIn("yolo11n-320.onnx", stdout.getvalue())

    def test_cli_rejects_invalid_imgsz_without_loading_ultralytics(self):
        completed = subprocess.run([sys.executable, str(ROOT / "export_model.py"), "yolo11n.pt", "--imgsz", "0"],
                                   cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("imgsz must be positive", completed.stderr)
        self.assertNotIn("ultralytics", completed.stderr.lower())

    def test_help_does_not_import_model_dependencies(self):
        completed = subprocess.run([sys.executable, str(ROOT / "export_model.py"), "--help"],
                                   cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--imgsz", completed.stdout)


if __name__ == "__main__":
    unittest.main()
