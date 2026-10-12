"""Pure helpers that convert VisDrone2019-DET annotations into a 6-class YOLO dataset.

VisDrone rows: ``bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion``.
``score == 0`` marks an ignored region. Categories 7, 8 (tricycles), 0 and 11 have no target class;
their pixels are masked instead of being kept as unlabeled background.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

CLASS_NAMES: list[str] = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

# VisDrone category id -> target class index.
VISDRONE_TO_TARGET: dict[int, int] = {
    1: 0,  # pedestrian
    2: 0,  # people
    3: 1,  # bicycle
    4: 2,  # car
    5: 2,  # van
    10: 3,  # motor
    9: 4,  # bus
    6: 5,  # truck
}

# Target class index -> COCO class index, for the zero-shot baseline.
COCO_CLASS_IDS: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7}

Box = tuple[int, int, int, int, int]  # (class, left, top, width, height) in pixels
Rect = tuple[int, int, int, int]  # (left, top, width, height) in pixels


def parse_annotation(text: str) -> tuple[list[Box], list[Rect]]:
    """Split one VisDrone annotation file into target boxes and regions to mask."""
    boxes: list[Box] = []
    masks: list[Rect] = []
    for line in text.splitlines():
        fields = [field for field in line.strip().split(",") if field != ""]
        if len(fields) < 6:
            continue
        left, top, width, height, score, category = (int(value) for value in fields[:6])
        if width <= 0 or height <= 0:
            continue
        target = VISDRONE_TO_TARGET.get(category)
        if score == 0 or target is None:
            masks.append((left, top, width, height))
        else:
            boxes.append((target, left, top, width, height))
    return boxes, masks


def to_yolo_lines(
    boxes: Iterable[Box], img_w: int, img_h: int, class_map: Mapping[int, int] | None = None
) -> list[str]:
    """Convert pixel boxes to normalized YOLO ``class xc yc w h`` lines, clipped to the image."""
    lines = []
    for cls, left, top, width, height in boxes:
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(img_w, left + width), min(img_h, top + height)
        if x1 <= x0 or y1 <= y0:
            continue
        label = class_map[cls] if class_map is not None else cls
        xc, yc = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
        w, h = (x1 - x0) / img_w, (y1 - y0) / img_h
        lines.append(f"{label} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def sequence_id(stem: str) -> str:
    """VisDrone image stems start with the id of the video sequence they were taken from."""
    return stem.split("_", 1)[0]


def sample_by_sequence(stems: Sequence[str], n: int, seed: int) -> list[str]:
    """Deterministically pick ``n`` images, round-robin across sequences to keep scene diversity."""
    if n >= len(stems):
        return list(stems)
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in sorted(stems):
        groups[sequence_id(stem)].append(stem)
    for members in groups.values():
        rng.shuffle(members)
    order = sorted(groups)
    rng.shuffle(order)
    picked: list[str] = []
    while len(picked) < n:
        for seq in order:
            if groups[seq] and len(picked) < n:
                picked.append(groups[seq].pop())
    return picked
