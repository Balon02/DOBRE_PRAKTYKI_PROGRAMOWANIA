import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
from typing import List, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from inference import InferencePipeline

try:
    from redislite import Redis
    from rq import Queue
    from rq.job import Job
    from rq.worker import SimpleWorker
except ImportError:  # pragma: no cover - optional in direct mode
    Redis = None
    Queue = None
    Job = None
    SimpleWorker = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("plate-api")


app = FastAPI(title="Plate Reader API")

PIPELINE = None


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode bytes into a BGR image using OpenCV."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def normalize_chunk_result(chunk_size: int, chunk_result) -> List[str]:
    """Normalize pipeline output to a list of strings matching the chunk length."""
    if chunk_result is None:
        return [""] * chunk_size
    if isinstance(chunk_result, str):
        return [chunk_result] + [""] * (chunk_size - 1)
    results: List[str] = []
    for i in range(chunk_size):
        if i < len(chunk_result):
            val = chunk_result[i]
            results.append(val if val else "")
        else:
            results.append("")
    return results


def run_pipeline_batched(pipeline: InferencePipeline, images: List[np.ndarray]) -> List[str]:
    """Run inference in fixed batches, padding handled inside the pipeline."""
    batch_size = pipeline.batch_size
    outputs: List[str] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        try:
            chunk_result = pipeline.infer(chunk)
        except Exception:
            logger.exception("Pipeline inference failed")
            outputs.extend([""] * len(chunk))
            continue
        outputs.extend(normalize_chunk_result(len(chunk), chunk_result))
    return outputs


def process_images_job(image_payloads: List[bytes]) -> List[str]:
    """RQ job: decode images and run through the shared pipeline."""
    global PIPELINE
    if PIPELINE is None:
        raise RuntimeError("Pipeline not initialized in worker process")

    results: List[str] = []
    valid_images: List[np.ndarray] = []
    valid_indices: List[int] = []
    for idx, payload in enumerate(image_payloads):
        img = decode_image_bytes(payload)
        if img is None:
            results.append("")
        else:
            results.append(None)  # placeholder
            valid_images.append(img)
            valid_indices.append(idx)

    if valid_images:
        preds = run_pipeline_batched(PIPELINE, valid_images)
        for local_idx, plate in enumerate(preds):
            global_idx = valid_indices[local_idx]
            results[global_idx] = plate if plate else ""

    for i, val in enumerate(results):
        if val is None:
            results[i] = ""
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="License plate API")
    parser.add_argument("--mode", choices=["direct", "queue", "worker"], default="direct")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--num-consumers",
        type=int,
        default=2,
        help="Number of worker processes to spawn in queue mode",
    )
    parser.add_argument(
        "--redis-path",
        default=".redislite.db",
        help="Path for redislite database/socket",
    )
    parser.add_argument(
        "--queue-name",
        default="plate-jobs",
        help="RQ queue name",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Uvicorn log level",
    )
    return parser


def start_redislite(path: str) -> Redis:
    if Redis is None:
        raise RuntimeError("redislite and rq are required for queue/worker modes. Please install them.")
    redis_conn = Redis(path)
    redis_conn.ping()
    logger.info("Started redislite at %s", path)
    return redis_conn


def launch_workers(args, redis_path: str) -> List[subprocess.Popen]:
    procs: List[subprocess.Popen] = []
    for i in range(args.num_consumers):
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            "worker",
            "--batch",
            str(args.batch),
            "--redis-path",
            redis_path,
            "--queue-name",
            args.queue_name,
        ]
        proc = subprocess.Popen(cmd)
        procs.append(proc)
        logger.info("Launched worker %s with PID %s", i + 1, proc.pid)
    return procs


def terminate_workers(procs: List[subprocess.Popen]):
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if getattr(app.state, "mode", None) != "direct":
        return JSONResponse(status_code=503, content={"status": "fail", "plates": []})
    if not files:
        return {"status": "fail", "plates": []}

    pipeline: InferencePipeline = app.state.pipeline
    results: List[str] = []
    valid_images: List[np.ndarray] = []
    valid_indices: List[int] = []
    had_any_valid = False

    for idx, upload in enumerate(files):
        data = await upload.read()
        img = decode_image_bytes(data)
        if img is None:
            results.append("")
        else:
            results.append(None)  # placeholder
            valid_images.append(img)
            valid_indices.append(idx)
            had_any_valid = True

    if valid_images:
        preds = await run_in_threadpool(run_pipeline_batched, pipeline, valid_images)
        for local_idx, plate in enumerate(preds):
            global_idx = valid_indices[local_idx]
            results[global_idx] = plate if plate else ""

    for i, val in enumerate(results):
        if val is None:
            results[i] = ""

    status = "ok" if had_any_valid else "fail"
    return {"status": status, "plates": results}


@app.post("/enqueue")
async def enqueue(files: List[UploadFile] = File(...)):
    if getattr(app.state, "mode", None) != "queue":
        return JSONResponse(status_code=503, content={"status": "fail", "job_id": None})
    if not files:
        return {"status": "fail", "job_id": None}

    payloads: List[bytes] = []
    for upload in files:
        data = await upload.read()
        payloads.append(data)

    queue: Queue = app.state.queue
    try:
        job = queue.enqueue(
            process_images_job,
            args=(payloads,),
            result_ttl=3600,
            failure_ttl=3600,
            job_timeout=600,
        )
    except Exception:
        logger.exception("Failed to enqueue job")
        return {"status": "fail", "job_id": None}

    return {"status": "ok", "job_id": job.id}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    if getattr(app.state, "mode", None) != "queue":
        return JSONResponse(status_code=503, content={"status": "fail", "plates": []})
    try:
        job = Job.fetch(job_id, connection=app.state.redis)
    except Exception:
        return {"status": "not_found", "plates": []}

    status = job.get_status(refresh=True)
    if status == "finished":
        return {"status": "ok", "plates": job.result or []}
    if status == "failed":
        return {"status": "fail", "plates": []}
    return {"status": status, "plates": []}


@app.get("/health")
async def health():
    return {"status": "ok", "mode": getattr(app.state, "mode", "unknown")}


def run_direct(args):
    logger.info("Starting in direct mode with batch=%s", args.batch)
    pipeline = InferencePipeline(batch=args.batch)
    app.state.mode = "direct"
    app.state.pipeline = pipeline
    app.state.batch = args.batch
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def run_queue(args):
    redis_conn = start_redislite(args.redis_path)
    queue = Queue(args.queue_name, connection=redis_conn)
    app.state.mode = "queue"
    app.state.queue = queue
    app.state.redis = redis_conn
    app.state.batch = args.batch

    workers = launch_workers(args, args.redis_path)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    finally:
        terminate_workers(workers)


def run_worker(args):
    global PIPELINE
    redis_conn = start_redislite(args.redis_path)
    queue = Queue(args.queue_name, connection=redis_conn)
    PIPELINE = InferencePipeline(batch=args.batch)
    worker = SimpleWorker([queue], connection=redis_conn, name=f"worker-b{args.batch}")
    logger.info("Worker started with batch=%s", args.batch)
    worker.work(with_scheduler=False, logging_level=logging.INFO)


def main():
    args = build_arg_parser().parse_args()
    if args.mode in ("queue", "worker") and Redis is None:
        raise SystemExit("Queue/worker modes require redislite and rq. Install with: pip install redislite redis rq")

    if args.mode == "direct":
        run_direct(args)
    elif args.mode == "queue":
        run_queue(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
