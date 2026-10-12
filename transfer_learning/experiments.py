"""Training and evaluation for the transfer-learning study.

Runs, all on the same data, image size, batch, seed and device:

* ``zeroshot``  - COCO-pretrained YOLO11n evaluated as-is (classes mapped to COCO ids).
* ``scratch``   - YOLO11n from random weights, ``stage1 + stage2`` epochs in one run.
* ``stage1``    - COCO weights, backbone frozen (modules 0-10), new 6-class head trained.
* ``stage2``    - starts from ``stage1`` best weights, everything unfrozen, lower LR, cosine decay.
* ``full_ft``   - ablation: COCO weights, single-stage full fine-tune with the same epoch budget.

Usage:
    .venv/bin/python -m transfer_learning.experiments all --stage1-epochs 10 --stage2-epochs 30
    .venv/bin/python -m transfer_learning.experiments eval --weights runs/act3/stage2/weights/best.pt --name stage2
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from transfer_learning.visdrone import CLASS_NAMES, COCO_CLASS_IDS

PROJECT = Path("runs/act3").resolve()
DATA = Path("data/visdrone6.yaml")
DATA_COCO = Path("data/visdrone6_coco.yaml")
CM_CONF = 0.25  # operating point for the confusion matrix
CM_IOU = 0.45  # Ultralytics ConfusionMatrix.process_batch matching IoU (default used by the detect validator)
BACKBONE_LAYERS = 11  # YOLO11 backbone = modules 0-10 (Conv/C3k2 ... SPPF, C2PSA)


def common_args(device: str, batch: int, imgsz: int, workers: int) -> dict[str, Any]:
    return {
        "data": str(DATA), "imgsz": imgsz, "batch": batch, "device": device, "workers": workers,
        "seed": 0, "deterministic": True, "project": str(PROJECT), "exist_ok": True, "plots": True,
        "optimizer": "AdamW", "weight_decay": 5e-4, "cache": "ram", "verbose": False,
    }


def _train(model_source: str, name: str, epochs: int, **overrides: Any) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_source)
    started = time.perf_counter()
    model.train(name=name, epochs=epochs, patience=epochs, **overrides)
    elapsed = time.perf_counter() - started
    info = {"name": name, "source": model_source, "epochs": epochs, "train_seconds": round(elapsed, 1),
            "params": sum(p.numel() for p in model.model.parameters()),
            "args": {k: v for k, v in overrides.items() if k not in {"project"}}}
    (PROJECT / name / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def train_scratch(epochs: int, base: dict[str, Any]) -> dict[str, Any]:
    return _train("yolo11n.yaml", "scratch", epochs, pretrained=False, lr0=1e-3, warmup_epochs=3, **base)


def train_stage1(epochs: int, base: dict[str, Any]) -> dict[str, Any]:
    return _train("yolo11n.pt", "stage1", epochs, freeze=BACKBONE_LAYERS, lr0=1e-3, warmup_epochs=1, **base)


def train_stage2(epochs: int, base: dict[str, Any]) -> dict[str, Any]:
    stage1 = PROJECT / "stage1" / "weights" / "best.pt"
    if not stage1.exists():
        raise FileNotFoundError(f"Stage 1 weights not found: {stage1}")
    return _train(str(stage1), "stage2", epochs, freeze=0, lr0=2e-4, lrf=0.05, cos_lr=True,
                  warmup_epochs=0, **base)


def train_full_ft(epochs: int, base: dict[str, Any]) -> dict[str, Any]:
    return _train("yolo11n.pt", "full_ft", epochs, freeze=0, lr0=1e-3, warmup_epochs=1, **base)


def evaluate(weights: str, name: str, device: str, batch: int, imgsz: int, zeroshot: bool = False) -> dict[str, Any]:
    """Evaluate on the held-out test split; export per-class metrics and the confusion matrix."""
    from ultralytics import YOLO

    model = YOLO(weights)
    out_dir = PROJECT / "eval" / name
    data = DATA_COCO if zeroshot else DATA
    class_ids = [COCO_CLASS_IDS[i] for i in range(len(CLASS_NAMES))] if zeroshot else list(range(len(CLASS_NAMES)))
    common = {"data": str(data), "split": "test", "imgsz": imgsz, "batch": batch, "device": device, "iou": 0.7,
              "project": str(PROJECT / "eval"), "exist_ok": True, "classes": class_ids if zeroshot else None,
              "verbose": False}
    # mAP needs the full precision-recall curve, so it is computed from (almost) every candidate box.
    metrics = model.val(conf=0.001, plots=True, name=name, **common)
    # The validator builds its confusion matrix at the same conf, which would flood it with low-confidence boxes;
    # a second pass at the deployment threshold gives an operating-point matrix.
    cm_metrics = model.val(conf=CM_CONF, plots=True, name=f"{name}_cm", **common)
    box = metrics.box
    per_class = {}
    for target, class_id in enumerate(class_ids):
        support = int(metrics.nt_per_class[class_id]) if metrics.nt_per_class is not None else 0
        if class_id in list(box.ap_class_index):
            p, r, ap50, ap = box.class_result(list(box.ap_class_index).index(class_id))
        else:
            p = r = ap50 = ap = 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[CLASS_NAMES[target]] = {"precision": p, "recall": r, "f1": f1, "map50": ap50, "map50_95": ap,
                                          "support": support}
    # Confusion matrix rows = predicted, cols = true; last index = background. Keep only our classes.
    matrix = cm_metrics.confusion_matrix.matrix
    keep = class_ids + [matrix.shape[0] - 1]
    confusion = matrix[np.ix_(keep, keep)].astype(int).tolist()
    result = {
        "name": name, "weights": weights, "zeroshot": zeroshot,
        "params": sum(p.numel() for p in model.model.parameters()),
        "map50": float(np.mean([v["map50"] for v in per_class.values()])),
        "map50_95": float(np.mean([v["map50_95"] for v in per_class.values()])),
        "precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "per_class": per_class, "confusion_matrix": {"labels": CLASS_NAMES + ["background"], "matrix": confusion,
                                                     "conf_threshold": CM_CONF, "iou_threshold": CM_IOU},
        "speed_ms_per_image": metrics.speed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def split_gap(names: tuple[str, ...], device: str, batch: int, imgsz: int) -> dict[str, Any]:
    """mAP/precision/recall of each model's best weights on train, val and test (overfitting check)."""
    from ultralytics import YOLO

    gap: dict[str, Any] = {}
    for name in names:
        model = YOLO(str(PROJECT / name / "weights" / "best.pt"))
        gap[name] = {}
        for split in ("train", "val", "test"):
            m = model.val(data=str(DATA), split=split, imgsz=imgsz, batch=batch, device=device, conf=0.001, iou=0.7,
                          plots=False, project=str(PROJECT / "eval"), name="_gap", exist_ok=True, verbose=False)
            gap[name][split] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
                                "recall": float(m.box.mr)}
    (PROJECT / "eval" / "split_gap.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
    shutil.rmtree(PROJECT / "eval" / "_gap", ignore_errors=True)
    return gap


def module_drift(weights_a: str, weights_b: str) -> list[dict[str, Any]]:
    """Per-module max |w_a - w_b| for modules shared by two checkpoints (proves what was frozen)."""
    import torch

    def state(path: str) -> dict[str, torch.Tensor]:
        from ultralytics import YOLO

        return {k: v.detach().float().cpu() for k, v in YOLO(path).model.state_dict().items()}

    a, b = state(weights_a), state(weights_b)
    drift: dict[int, dict[str, Any]] = {}
    for key, tensor in a.items():
        if key not in b or b[key].shape != tensor.shape or not key.startswith("model."):
            continue
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue  # BatchNorm statistics are buffers, not trainable weights
        index = int(key.split(".")[1])
        entry = drift.setdefault(index, {"module": index, "max_abs_diff": 0.0, "params": 0})
        entry["max_abs_diff"] = max(entry["max_abs_diff"], float((tensor - b[key]).abs().max()))
        entry["params"] += tensor.numel()
    return [drift[i] for i in sorted(drift)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["all", "zeroshot", "scratch", "stage1", "stage2", "full_ft", "eval"])
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights", help="weights for the eval step")
    parser.add_argument("--name", help="output name for the eval step")
    args = parser.parse_args(argv)

    base = common_args(args.device, args.batch, args.imgsz, args.workers)
    total = args.stage1_epochs + args.stage2_epochs
    ev = {"device": args.device, "batch": args.batch, "imgsz": args.imgsz}
    steps = ["zeroshot", "stage1", "stage2", "scratch", "full_ft"] if args.step == "all" else [args.step]
    for step in steps:
        if step == "eval":
            if not args.weights or not args.name:
                parser.error("eval needs --weights and --name")
            evaluate(args.weights, args.name, **ev)
        elif step == "zeroshot":
            evaluate("yolo11n.pt", "zeroshot", zeroshot=True, **ev)
        else:
            trainer = {"scratch": (train_scratch, total), "stage1": (train_stage1, args.stage1_epochs),
                       "stage2": (train_stage2, args.stage2_epochs), "full_ft": (train_full_ft, total)}[step]
            trainer[0](trainer[1], base)
            evaluate(str(PROJECT / step / "weights" / "best.pt"), step, **ev)
    shutil.rmtree(Path("runs/act3/eval/_tmp"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
