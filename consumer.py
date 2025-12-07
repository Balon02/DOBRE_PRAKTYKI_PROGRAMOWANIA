from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from queue_utils import (
    QUEUE_FILENAME,
    claim_next_task,
    mark_done,
)


def process_task(queue_path: Path, consumer_id: str, task_id: str, processing_seconds: int) -> None:
    print(f"[{consumer_id}] Starting task {task_id} (processing {processing_seconds}s)")
    time.sleep(processing_seconds)
    mark_done(queue_path, task_id)
    print(f"[{consumer_id}] Finished task {task_id}")


def consume(queue_path: Path, consumer_id: str, processing_seconds: int, poll_seconds: int, exit_when_empty: bool) -> None:
    while True:
        task = claim_next_task(queue_path, consumer_id)
        if task is None:
            if exit_when_empty:
                print(f"[{consumer_id}] No tasks left, exiting.")
                return
            time.sleep(poll_seconds)
            continue
        process_task(queue_path, consumer_id, task.task_id, processing_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue consumer (SQLite)")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(__file__).resolve().parent / QUEUE_FILENAME,
        help="Path to SQLite queue file",
    )
    parser.add_argument(
        "--processing-seconds",
        type=int,
        default=int(os.environ.get("PROCESSING_SECONDS", 30)),
        help="Seconds to spend on each task (default 30)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=5,
        help="Seconds between queue checks when idle",
    )
    parser.add_argument(
        "--exit-when-empty",
        action="store_true",
        help="Exit once queue has no pending tasks",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=f"consumer-{os.getpid()}",
        help="Identifier for this consumer",
    )
    args = parser.parse_args()

    consume(
        queue_path=args.queue,
        consumer_id=args.name,
        processing_seconds=args.processing_seconds,
        poll_seconds=args.poll_seconds,
        exit_when_empty=args.exit_when_empty,
    )


if __name__ == "__main__":
    main()
