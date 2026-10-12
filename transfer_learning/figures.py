"""Tables and figures built only from the exported run artifacts (results.csv, metrics.json)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path("runs/act3")
MODELS = ["zeroshot", "scratch", "stage1", "stage2", "full_ft"]
LABELS = {
    "zeroshot": "COCO sin ajuste",
    "scratch": "Desde cero",
    "stage1": "Etapa 1 (backbone congelado)",
    "stage2": "Etapa 2 (fine-tuning, propuesto)",
    "full_ft": "Fine-tuning en una etapa",
}
# Fixed categorical order (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6 on #fcfcfb).
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_eval(name: str, project: Path = PROJECT) -> dict[str, Any] | None:
    path = project / "eval" / name / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_curve(name: str, project: Path = PROJECT) -> dict[str, np.ndarray] | None:
    path = project / name / "results.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(fh)]
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def load_train_info(name: str, project: Path = PROJECT) -> dict[str, Any] | None:
    path = project / name / "train_info.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def comparison_rows(project: Path = PROJECT) -> list[dict[str, Any]]:
    rows = []
    for name in MODELS:
        ev = load_eval(name, project)
        if ev is None:
            continue
        info = load_train_info(name, project) or {}
        curve = load_curve(name, project)
        best_epoch = int(curve["epoch"][int(np.argmax(curve["metrics/mAP50-95(B)"]))]) if curve else None
        rows.append({
            "model": LABELS[name], "key": name, "map50": ev["map50"], "map50_95": ev["map50_95"],
            "precision": ev["precision"], "recall": ev["recall"], "params_m": ev["params"] / 1e6,
            "epochs": info.get("epochs", 0), "best_epoch": best_epoch,
            "train_min": info.get("train_seconds", 0) / 60, "inference_ms": ev["speed_ms_per_image"]["inference"],
        })
    if "stage2" in {r["key"] for r in rows} and "stage1" in {r["key"] for r in rows}:
        stage1 = next(r for r in rows if r["key"] == "stage1")
        stage2 = next(r for r in rows if r["key"] == "stage2")
        stage2["epochs"] += stage1["epochs"]
        stage2["train_min"] += stage1["train_min"]
    return rows


def plot_training_curves(out: Path, project: Path = PROJECT) -> Path:
    """Train (dashed) vs validation (solid) losses, plus validation precision and mAP@50, per epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    stage1, stage2 = load_curve("stage1", project), load_curve("stage2", project)
    series = {}
    if stage1 is not None and stage2 is not None:
        offset = stage1["epoch"][-1]
        series["stage2"] = {k: np.concatenate([stage1[k], stage2[k]]) for k in stage1 if k in stage2}
        series["stage2"]["epoch"] = np.concatenate([stage1["epoch"], stage2["epoch"] + offset])
    for name in ("scratch", "full_ft"):
        curve = load_curve(name, project)
        if curve is not None:
            series[name] = curve
    panels = ((axes[0], ("train/cls_loss", "val/cls_loss"), "Pérdida de clasificación (BCE)"),
              (axes[1], ("train/box_loss", "val/box_loss"), "Pérdida de caja (CIoU)"),
              (axes[2], ("metrics/precision(B)", "metrics/mAP50(B)"), "Validación: precisión y mAP@50"))
    for ax, (key_a, key_b), title in panels:
        for name, curve in series.items():
            ax.plot(curve["epoch"], curve[key_a], color=COLORS[name], linewidth=1.4, linestyle="--")
            ax.plot(curve["epoch"], curve[key_b], color=COLORS[name], linewidth=2)
        if stage1 is not None:
            ax.axvline(stage1["epoch"][-1] + 0.5, color=INK_2, linestyle=":", linewidth=1)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("época")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[n], linewidth=2,
                      label="Dos etapas (propuesto)" if n == "stage2" else LABELS[n]) for n in series]
    handles += [Line2D([], [], color=INK_2, linestyle="--", label="train / precisión"),
                Line2D([], [], color=INK_2, label="val / mAP@50"),
                Line2D([], [], color=INK_2, linestyle=":", label="descongelado (ép. 10)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_model_comparison(out: Path, project: Path = PROJECT) -> Path:
    rows = comparison_rows(project)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    width = 0.38
    x = np.arange(len(rows))
    for offset, key, alpha, label in ((-width / 2, "map50", 1.0, "mAP@50"), (width / 2, "map50_95", 0.55, "mAP@50-95")):
        bars = ax.bar(x + offset, [r[key] * 100 for r in rows], width - 0.04,
                      color=[COLORS[r["key"]] for r in rows], alpha=alpha, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=7.5, color=INK, padding=2)
    ax.set_xticks(x, [r["model"] for r in rows], fontsize=8)
    ax.set_ylabel("% en test")
    ax.set_title("mAP en test por modelo (sólido: mAP@50, claro: mAP@50-95)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_per_class(out: Path, names: tuple[str, ...] = ("zeroshot", "scratch", "stage2"), project: Path = PROJECT) -> Path:
    evals = {n: load_eval(n, project) for n in names}
    evals = {n: e for n, e in evals.items() if e is not None}
    classes = list(next(iter(evals.values()))["per_class"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    width = 0.8 / len(evals)
    x = np.arange(len(classes))
    for i, (name, ev) in enumerate(evals.items()):
        bars = ax.bar(x + (i - (len(evals) - 1) / 2) * width, [ev["per_class"][c]["map50"] * 100 for c in classes],
                      width - 0.03, color=COLORS[name], label=LABELS[name])
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=INK, padding=1)
    ax.set_xticks(x, classes)
    ax.set_ylabel("AP@50 (%)")
    ax.set_title("AP@50 por clase en test", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=len(evals), loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def normalized_confusion(ev: dict[str, Any]) -> np.ndarray:
    """Column-normalize: each true class (column) sums to 1; background column left as raw share of FPs."""
    matrix = np.array(ev["confusion_matrix"]["matrix"], dtype=float)
    totals = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def plot_confusion(out: Path, names: tuple[str, ...] = ("zeroshot", "stage2"), project: Path = PROJECT) -> Path:
    evals = [(n, load_eval(n, project)) for n in names]
    evals = [(n, e) for n, e in evals if e is not None]
    fig, axes = plt.subplots(1, len(evals), figsize=(3.6 * len(evals), 3.3), squeeze=False)
    for ax, (name, ev) in zip(axes[0], evals):
        norm = normalized_confusion(ev)
        labels = ev["confusion_matrix"]["labels"]
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for (r, c), value in np.ndenumerate(norm):
            if value >= 0.005:
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 else INK)
        ticks = [lab if lab != "background" else "fondo" for lab in labels]
        ax.set_xticks(range(len(labels)), ticks, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(labels)), ticks, fontsize=7.5)
        ax.set_xlabel("clase real")
        ax.set_ylabel("clase predicha")
        ax.set_title(LABELS[name], loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
