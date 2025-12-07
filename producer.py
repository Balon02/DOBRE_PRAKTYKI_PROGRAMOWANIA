from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from queue_utils import QUEUE_FILENAME, append_tasks


def generate_task_ids(count: int) -> list[str]:
    return [str(uuid.uuid4()) for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue producer (SQLite)")
    parser.add_argument("--count", type=int, default=1, help="How many tasks to enqueue")
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(__file__).resolve().parent / QUEUE_FILENAME,
        help="Path to SQLite queue file",
    )
    args = parser.parse_args()

    task_ids = generate_task_ids(args.count)
    append_tasks(args.queue, task_ids)
    print(f"Enqueued {len(task_ids)} tasks into {args.queue}")


if __name__ == "__main__":
    main()
