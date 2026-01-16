import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import requests


DATASET_DIR = Path("DATASET")
ANNOTATIONS_PATH = DATASET_DIR / "annotations.xml"
PHOTOS_DIR = DATASET_DIR / "photos"


def load_annotations() -> Dict[str, Tuple[str, str, str]]:
    """
    Returns mapping of image number (stem without extension) to tuple:
    (image_id, filename, plate_number).
    """
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()
    mapping: Dict[str, Tuple[str, str, str]] = {}
    for image in root.findall("image"):
        img_id = image.get("id")
        name = image.get("name")
        if not name:
            continue
        stem = Path(name).stem
        box = image.find("box")
        plate_attr = None
        if box is not None:
            plate_attr = box.find("attribute[@name='plate number']")
        plate = plate_attr.text.strip() if plate_attr is not None and plate_attr.text else ""
        mapping[stem] = (img_id, name, plate)
    return mapping


def get_server_mode(base_url: str, timeout: float) -> str | None:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        if resp.ok:
            data = resp.json()
            return data.get("mode")
    except Exception:
        return None
    return None


def post_predict(base_url: str, file_paths: List[Path], timeout: float) -> List[str] | None:
    files = []
    handles = []
    try:
        for path in file_paths:
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "application/octet-stream")))
        resp = requests.post(f"{base_url}/predict", files=files, timeout=timeout)
        if not resp.ok:
            return None
        data = resp.json()
        if data.get("status") != "ok":
            return None
        return data.get("plates", [])
    finally:
        for h in handles:
            h.close()


def enqueue_job(base_url: str, file_paths: List[Path], timeout: float) -> str | None:
    files = []
    handles = []
    try:
        for path in file_paths:
            handle = path.open("rb")
            handles.append(handle)
            files.append(("files", (path.name, handle, "application/octet-stream")))
        resp = requests.post(f"{base_url}/enqueue", files=files, timeout=timeout)
        if not resp.ok:
            return None
        data = resp.json()
        if data.get("status") != "ok":
            return None
        return data.get("job_id")
    finally:
        for h in handles:
            h.close()


def fetch_result(base_url: str, job_id: str, timeout: float, max_wait: float) -> List[str] | None:
    start = time.perf_counter()
    while True:
        resp = requests.get(f"{base_url}/result/{job_id}", timeout=timeout)
        if not resp.ok:
            return None
        data = resp.json()
        status = data.get("status")
        if status == "ok":
            return data.get("plates", [])
        if status == "fail" or status == "not_found":
            return None
        if (time.perf_counter() - start) > max_wait:
            return None
        time.sleep(0.2)


def evaluate(
    base_url: str,
    ids: List[str],
    send_mode: str,
    timeout: float,
    max_wait: float,
) -> None:
    annotations = load_annotations()
    samples: List[Tuple[str, Path, str]] = []
    for img_num in ids:
        if img_num not in annotations:
            print(f"Image {img_num} not found in annotations. Skipping.")
            continue
        _, filename, plate = annotations[img_num]
        path = PHOTOS_DIR / filename
        if not path.exists():
            print(f"Image file missing: {path}. Skipping.")
            continue
        samples.append((img_num, path, plate))

    if not samples:
        print("No valid samples to evaluate.")
        return

    mode = get_server_mode(base_url, timeout)
    use_queue = mode == "queue"

    total = len(samples)
    correct = 0
    failures: List[Tuple[str, str, str]] = []  # img_num, expected, predicted

    start_time = time.perf_counter()

    def process_batch(batch_items: List[Tuple[str, Path, str]]):
        nonlocal correct
        file_paths = [p for (_, p, _) in batch_items]
        expected = [plate for (_, _, plate) in batch_items]
        preds: List[str] | None = None
        if use_queue:
            job_id = enqueue_job(base_url, file_paths, timeout)
            if job_id:
                preds = fetch_result(base_url, job_id, timeout, max_wait)
        else:
            preds = post_predict(base_url, file_paths, timeout)

        if not preds:
            for (img_num, _, exp) in batch_items:
                failures.append((img_num, exp, ""))
            return

        if len(preds) != len(batch_items):
            # treat mismatch as failures
            for (img_num, _, exp) in batch_items:
                failures.append((img_num, exp, ""))
            return

        for (img_num, _, exp), pred in zip(batch_items, preds):
            exp_clean = exp.strip()
            pred_clean = pred.strip() if isinstance(pred, str) else ""
            if pred_clean == exp_clean:
                correct += 1
            else:
                failures.append((img_num, exp, pred_clean))

    if send_mode == "batch":
        process_batch(samples)
    else:
        for sample in samples:
            process_batch([sample])

    total_time = time.perf_counter() - start_time
    accuracy = correct / total if total else 0.0
    per_image = total_time / total if total else 0.0

    print(f"Processed images: {total}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Total time: {total_time:.3f}s")
    print(f"Avg time per image: {per_image:.3f}s")
    if failures:
        print("Failures (image_num, expected, predicted):")
        for img_num, exp, pred in failures:
            print(f"  {img_num}: expected={exp}, predicted={pred}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate plate API against dataset annotations.")
    parser.add_argument(
        "ids",
        nargs="+",
        help="Image numbers (stem without extension) to evaluate, e.g. 1 10 100",
    )
    parser.add_argument(
        "--send-mode",
        choices=["batch", "sequential"],
        default="batch",
        help="Send all images in one request/job or one-by-one.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the running API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=180.0,
        help="Max wait time for queue job completion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(
        base_url=args.api_url.rstrip("/"),
        ids=[str(i) for i in args.ids],
        send_mode=args.send_mode,
        timeout=args.timeout,
        max_wait=args.max_wait,
    )


if __name__ == "__main__":
    main()
