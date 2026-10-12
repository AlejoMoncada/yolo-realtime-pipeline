"""Build the 6-class VisDrone YOLO dataset straight from the official zips.

Images are read from the zip archives (no full extraction, so it fits on a small disk),
ignored/unmapped regions are filled with gray, and images are downscaled to ``--max-side``.
The official VisDrone splits come from disjoint video sequences, so train/val/test never
share frames from the same clip. The train split is subsampled round-robin by sequence.

Usage:
    .venv/bin/python -m transfer_learning.prepare_data --download --delete-zips
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from transfer_learning.visdrone import (
    CLASS_NAMES,
    COCO_CLASS_IDS,
    parse_annotation,
    sample_by_sequence,
    to_yolo_lines,
)

SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
ZIP_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
MASK_VALUE = 114  # Ultralytics letterbox gray


def _members(archive: zipfile.ZipFile, prefix: str) -> dict[str, tuple[str, str]]:
    names = set(archive.namelist())
    pairs = {}
    for name in names:
        if name.endswith(".txt") and "/annotations/" in name:
            stem = Path(name).stem
            image = f"{prefix}/images/{stem}.jpg"
            if image in names:
                pairs[stem] = (image, name)
    return pairs


def convert_split(zip_path: Path, prefix: str, split: str, out: Path, coco_out: Path | None,
                  limit: int | None, max_side: int, seed: int) -> dict:
    images_dir, labels_dir = out / "images" / split, out / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if coco_out is not None:
        (coco_out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    masked = 0
    with zipfile.ZipFile(zip_path) as archive:
        pairs = _members(archive, prefix)
        stems = sample_by_sequence(sorted(pairs), limit, seed) if limit else sorted(pairs)
        for stem in stems:
            image_name, label_name = pairs[stem]
            image = cv2.imdecode(np.frombuffer(archive.read(image_name), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image in {zip_path.name}: {image_name}")
            height, width = image.shape[:2]
            boxes, masks = parse_annotation(archive.read(label_name).decode("utf-8"))
            for left, top, w, h in masks:
                image[max(0, top):max(0, top + h), max(0, left):max(0, left + w)] = MASK_VALUE
            masked += len(masks)
            scale = max_side / max(height, width)
            if scale < 1:
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = to_yolo_lines(boxes, width, height)
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if coco_out is not None:
                coco_lines = to_yolo_lines(boxes, width, height, class_map=COCO_CLASS_IDS)
                (coco_out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(coco_lines) + ("\n" if coco_lines else ""), encoding="utf-8")
            counts.update(CLASS_NAMES[int(line.split()[0])] for line in lines)
    return {"images": len(stems), "available": len(pairs), "masked_regions": masked,
            "instances": {name: counts.get(name, 0) for name in CLASS_NAMES}}


def write_yamls(out: Path, coco_out: Path) -> None:
    import ultralytics

    coco_names = yaml.safe_load(
        (Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml").read_text(encoding="utf-8"))["names"]
    common = {"train": "images/train", "val": "images/val", "test": "images/test"}
    (out.parent / f"{out.name}.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), **common, "names": dict(enumerate(CLASS_NAMES))}, sort_keys=False))
    # Same test images, labels expressed with COCO ids, for the zero-shot pretrained baseline.
    # Hard links, not a directory symlink: Ultralytics resolves symlinks and would then read the 6-class labels.
    images = coco_out / "images" / "test"
    if images.is_symlink():
        images.unlink()
    images.mkdir(parents=True, exist_ok=True)
    for source in sorted((out / "images" / "test").glob("*.jpg")):
        target = images / source.name
        if not target.exists():
            os.link(source, target)
    (coco_out.parent / f"{coco_out.name}.yaml").write_text(yaml.safe_dump(
        {"path": str(coco_out.resolve()), "train": "images/test", "val": "images/test", "test": "images/test",
         "names": coco_names}, sort_keys=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/visdrone6"))
    parser.add_argument("--train-images", type=int, default=2000, help="train subset size (sequence round-robin)")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="download missing official zips (~1.9 GB)")
    parser.add_argument("--delete-zips", action="store_true", help="delete each zip after converting it")
    args = parser.parse_args(argv)

    coco_out = args.out.parent / f"{args.out.name}_coco"
    stats = {}
    for split, prefix in SPLITS.items():
        zip_path = args.raw_dir / f"{prefix}.zip"
        if not zip_path.exists():
            if not args.download:
                parser.error(f"{zip_path} not found; pass --download to fetch it")
            from ultralytics.utils.downloads import download

            download(f"{ZIP_URL}/{prefix}.zip", dir=args.raw_dir, unzip=False, retry=5)
        limit = args.train_images if split == "train" else None
        stats[split] = convert_split(zip_path, prefix, split, args.out, coco_out if split == "test" else None,
                                     limit, args.max_side, args.seed)
        print(split, json.dumps(stats[split]))
        if args.delete_zips:
            zip_path.unlink()
    write_yamls(args.out, coco_out)
    stats["config"] = {"train_images": args.train_images, "max_side": args.max_side, "seed": args.seed,
                       "classes": CLASS_NAMES}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
