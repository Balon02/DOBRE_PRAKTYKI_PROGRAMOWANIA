#!/usr/bin/env python3
import argparse
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from grade import calculate_final_grade
from inference import InferencePipeline

warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recreate CVAT XML split (seeded) and load images with OpenCV."
    )
    parser.add_argument(
        "--annot_file",
        type=Path,
        default=Path("archive/annotations.xml"),
    )
    parser.add_argument(
        "--img_root",
        type=Path,
        default=Path("archive/photos"),
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.3,
        help="Fraction for validation (default 0.3 => 70/30 split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Shuffle seed for reproducible splits",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size (default 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Thread count for OpenCV loads (0 uses CPU count)",
    )
    return parser.parse_args()


PLATE_ATTR_PRIORITY = (
    "license_plate",
    "licence_plate",
    "plate_number",
    "plate",
    "lp",
    "lpn",
    "number",
    "text",
)


def _extract_plate_text(image_elem: ET.Element) -> str | None:
    attrs = []
    for attr in image_elem.findall(".//attribute"):
        name = attr.attrib.get("name", "").strip().lower()
        value = (attr.text or "").strip()
        if value:
            attrs.append((name, value))
    if not attrs:
        return None
    for preferred in PLATE_ATTR_PRIORITY:
        for name, value in attrs:
            if name == preferred:
                return value
    return attrs[0][1]


def load_entries(annot_file: Path):
    root = ET.parse(annot_file).getroot()
    images = root.findall(".//image")
    entries = []
    for img in images:
        name = img.attrib["name"]
        plate = _extract_plate_text(img)
        entries.append((name, plate))
    return entries


def make_split(entry_count: int, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(entry_count)
    rng.shuffle(indices)
    val_count = int(len(indices) * val_split)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return train_idx, val_idx


def _read_one(path: Path, flags: int):
    img = cv2.imread(os.fspath(path), flags)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def load_images_cv2(paths, executor: ThreadPoolExecutor | None, num_workers: int):
    flags = cv2.IMREAD_COLOR | getattr(cv2, "IMREAD_IGNORE_ORIENTATION", 0)
    read_one = partial(_read_one, flags=flags)
    if executor is not None:
        return list(executor.map(read_one, paths))
    return [_read_one(p, flags) for p in paths]


def _normalize_plate(text: str | None) -> str | None:
    if text is None:
        return None
    return text.strip().replace("_", "")


def _batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_split(
    name: str,
    pipeline: InferencePipeline,
    items: list[tuple[Path, str | None]],
    batch_size: int,
    executor: ThreadPoolExecutor | None,
    num_workers: int,
):
    total = 0
    correct = 0
    skipped = 0

    for batch in _batched(items, batch_size):
        paths, labels = zip(*batch)
        images = load_images_cv2(paths, executor, num_workers)
        preds = pipeline.infer(images)
        if preds is None:
            preds = [None] * len(images)
        for gt, pred in zip(labels, preds):
            gt_norm = _normalize_plate(gt)
            pred_norm = _normalize_plate(pred)
            if not gt_norm:
                skipped += 1
                continue
            total += 1
            if pred_norm == gt_norm:
                correct += 1

    acc = (correct / total) if total else 0.0
    print(
        f"{name} accuracy: {acc:.4f} ({correct}/{total}), skipped: {skipped}"
    )
    return acc


def measure_speed(
    pipeline: InferencePipeline,
    images: list[np.ndarray],
    batch_size: int,
):
    start = time.perf_counter()
    for batch in _batched(images, batch_size):
        pipeline.infer(batch)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    args = parse_args()
    entries = load_entries(args.annot_file)

    train_idx, val_idx = make_split(len(entries), args.val_split, args.seed)
    train_items = [(args.img_root / entries[i][0], entries[i][1]) for i in train_idx]
    val_items = [(args.img_root / entries[i][0], entries[i][1]) for i in val_idx]

    missing = [p for p, _ in (train_items + val_items) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing images (first 5): {missing[:5]}")

    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = os.cpu_count() or 1

    pipeline = InferencePipeline(batch=args.batch_size)

    executor = None
    if num_workers > 1:
        executor = ThreadPoolExecutor(max_workers=num_workers)
    try:
        train_acc = evaluate_split(
            "train",
            pipeline,
            train_items,
            args.batch_size,
            executor,
            num_workers,
        )
        val_acc = evaluate_split(
            "val",
            pipeline,
            val_items,
            args.batch_size,
            executor,
            num_workers,
        )

        rng = np.random.default_rng(args.seed)
        sample_count = min(100, len(entries))
        sample_idx = rng.choice(len(entries), size=sample_count, replace=False)
        sample_paths = [args.img_root / entries[i][0] for i in sample_idx]
        sample_images = load_images_cv2(sample_paths, executor, num_workers)

        elapsed = measure_speed(pipeline, sample_images, args.batch_size)
        per_100 = elapsed * (100.0 / sample_count) if sample_count else 0.0
        img_per_sec = (sample_count / elapsed) if elapsed > 0 else 0.0
        print(
            f"Speed: {per_100:.4f}s per 100 images "
            f"({img_per_sec:.2f} img/s, {sample_count} samples)"
        )
        train_grade = calculate_final_grade(train_acc * 100.0, per_100)
        val_grade = calculate_final_grade(val_acc * 100.0, per_100)
        print(f"Train grade: {train_grade:.1f}")
        print(f"Val grade: {val_grade:.1f}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
