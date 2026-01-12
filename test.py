import asyncio
import os
import random
import time
import xml.etree.ElementTree as ET

import torch

from main import DATASET_INFO, InferencePipeline


IOU_THRESHOLD = 0.5
SAMPLE_SIZE = 100
RANDOM_SEED = 1337


def _parse_cvat_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = []

    for image_el in root.findall("image"):
        name = image_el.get("name")
        if not name:
            continue
        boxes = []
        for box_el in image_el.findall("box"):
            xtl = float(box_el.get("xtl", 0))
            ytl = float(box_el.get("ytl", 0))
            xbr = float(box_el.get("xbr", 0))
            ybr = float(box_el.get("ybr", 0))
            boxes.append((xtl, ytl, xbr, ybr))
        if boxes:
            images.append({"name": name, "boxes": boxes})
    return images


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _best_iou(pred_box, gt_boxes):
    return max(_iou(pred_box, gt) for gt in gt_boxes)


def _image_path_from_xml(xml_path, image_name):
    base_dir = os.path.dirname(xml_path)
    return os.path.join(base_dir, image_name)


async def _run_eval():
    annotations = _parse_cvat_annotations(DATASET_INFO)
    if not annotations:
        raise RuntimeError("No annotated images found in CVAT XML.")

    rng = random.Random(RANDOM_SEED)
    sample = annotations
    if len(sample) > SAMPLE_SIZE:
        sample = rng.sample(annotations, SAMPLE_SIZE)

    pipeline = InferencePipeline()

    correct = 0
    timings = []

    for entry in sample:
        image_path = _image_path_from_xml(DATASET_INFO, entry["name"])
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        start = time.perf_counter()
        await pipeline.infer(image_bytes)
        timings.append(time.perf_counter() - start)

        with torch.inference_mode():
            preds = pipeline._detector.predict(
                source=image_path,
                imgsz=pipeline._detector_input[:2],
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False,
            )

        boxes = preds[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        best = boxes.conf.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best].tolist()
        if _best_iou((x1, y1, x2, y2), entry["boxes"]) >= IOU_THRESHOLD:
            correct += 1

    accuracy = correct / len(sample)
    avg_time = sum(timings) / len(timings)
    p95_time = sorted(timings)[int(0.95 * (len(timings) - 1))]

    print(f"Samples: {len(sample)}")
    print(f"Accuracy (IoU>={IOU_THRESHOLD}): {accuracy:.4f}")
    print(f"Avg inference time: {avg_time*1000:.2f} ms")
    print(f"P95 inference time: {p95_time*1000:.2f} ms")


if __name__ == "__main__":
    asyncio.run(_run_eval())

