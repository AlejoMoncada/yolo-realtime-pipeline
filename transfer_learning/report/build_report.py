"""Build the Act3 technical report (HTML -> PDF) from the run artifacts; no number is typed by hand.

Usage:
    .venv/bin/python -m transfer_learning.report.build_report
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from transfer_learning import experiments as ex
from transfer_learning import figures as fg
from transfer_learning.report import narrative

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",")


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def collect() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    ctx = {
        "stats": json.loads(Path("data/visdrone6/stats.json").read_text()),
        "rows": fg.comparison_rows(),
        "evals": {n: fg.load_eval(n) for n in fg.MODELS + fg.ABLATIONS[1:]},
        "gap": json.loads((ex.PROJECT / "eval" / "split_gap.json").read_text()),
        "drift": json.loads((ex.PROJECT / "eval" / "module_drift.json").read_text()),
        "curves": {n: fg.load_curve(n) for n in ("stage1", "stage2", "scratch", "full_ft") + tuple(fg.ABLATIONS[1:])},
        "sizes": json.loads((ex.PROJECT / "eval" / "object_sizes_test.json").read_text()),
        "info": {n: fg.load_train_info(n) for n in ("stage1", "stage2", "scratch", "full_ft")},
        "figs": {
            "curves": fg.plot_training_curves(FIG / "curves.png"),
            "comparison": fg.plot_model_comparison(FIG / "comparison.png"),
            "per_class": fg.plot_per_class(FIG / "per_class.png"),
            "confusion": fg.plot_confusion(FIG / "confusion.png"),
            "ablation": fg.plot_ablation(FIG / "ablation.png"),
        },
    }
    import torch
    import ultralytics

    ctx["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                       "ultralytics": ultralytics.__version__, "machine": "Apple M4 (MPS), 16 GB RAM, macOS"}
    cm = ctx["evals"]["stage2"]["confusion_matrix"]
    matrix = np.array(cm["matrix"])
    labels = cm["labels"]
    k = len(labels) - 1
    ctx["cm"] = {
        "labels": labels, "matrix": matrix,
        "missed": {labels[c]: int(matrix[k, c]) for c in range(k)},
        "true_total": {labels[c]: int(matrix[:, c].sum()) for c in range(k)},
        "fp_bg": {labels[r]: int(matrix[r, k]) for r in range(k)},
        "pairs": sorted(((int(matrix[r, c]), labels[r], labels[c]) for r in range(k) for c in range(k) if r != c),
                        reverse=True),
    }
    return ctx


def render(ctx: dict) -> str:
    css = (HERE / "report.css").read_text(encoding="utf-8")
    body = narrative.body(ctx, pct=pct, num=num, thousands=thousands)
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe Actividad 3 - Grupo 9</title><style>{css}</style></head><body>{body}</body></html>"""


def main() -> int:
    ctx = collect()
    html = HERE / "Informe_Actividad3_G9.html"
    html.write_text(render(ctx), encoding="utf-8")
    pdf = HERE / "Informe_Actividad3_G9.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    from pypdf import PdfReader

    pages = len(PdfReader(str(pdf)).pages)
    print(f"{pdf} pages={pages}")
    return 0 if pages <= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
