from __future__ import annotations

import csv
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

QUEUE_FILENAME = "queue.csv"
FIELDNAMES = ["task_id", "status", "created_at", "updated_at", "claimed_by"]
DEFAULT_STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"


@dataclass
class Task:
    task_id: str
    status: str
    created_at: str
    updated_at: str
    claimed_by: str

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Task":
        return cls(
            task_id=data["task_id"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            claimed_by=data["claimed_by"],
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "claimed_by": self.claimed_by,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked_file(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open(mode, newline="")
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        yield f
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def ensure_queue_file(path: Path) -> None:
    if not path.exists():
        with locked_file(path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_tasks(path: Path, task_ids: Iterable[str]) -> None:
    ensure_queue_file(path)
    now = _utc_now_iso()
    rows = [
        {
            "task_id": task_id,
            "status": DEFAULT_STATUS_PENDING,
            "created_at": now,
            "updated_at": now,
            "claimed_by": "",
        }
        for task_id in task_ids
    ]
    with locked_file(path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for row in rows:
            writer.writerow(row)


def _read_tasks(f) -> List[Task]:
    f.seek(0)
    reader = csv.DictReader(f)
    return [Task.from_dict(row) for row in reader]


def _write_tasks(f, tasks: List[Task]) -> None:
    f.seek(0)
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows([task.to_dict() for task in tasks])
    f.truncate()


def claim_next_task(path: Path, consumer_id: str) -> Optional[Task]:
    ensure_queue_file(path)
    with locked_file(path, "r+") as f:
        tasks = _read_tasks(f)
        pending = next((t for t in tasks if t.status == DEFAULT_STATUS_PENDING), None)
        if pending is None:
            return None
        pending.status = STATUS_IN_PROGRESS
        pending.claimed_by = consumer_id
        pending.updated_at = _utc_now_iso()
        _write_tasks(f, tasks)
        return pending


def mark_done(path: Path, task_id: str) -> bool:
    ensure_queue_file(path)
    with locked_file(path, "r+") as f:
        tasks = _read_tasks(f)
        updated = False
        for task in tasks:
            if task.task_id == task_id:
                task.status = STATUS_DONE
                task.updated_at = _utc_now_iso()
                updated = True
                break
        if updated:
            _write_tasks(f, tasks)
        return updated
