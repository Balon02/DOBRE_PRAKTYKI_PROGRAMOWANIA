from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

QUEUE_FILENAME = "queue.sqlite"
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"


@dataclass
class Task:
    id: int
    task_id: str
    status: str
    created_at: str
    updated_at: str
    claimed_by: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _get_conn(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Path) -> None:
    with _get_conn(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                claimed_by TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        conn.commit()


def append_tasks(path: Path, task_ids: Iterable[str]) -> None:
    init_db(path)
    now = _utc_now_iso()
    rows = [(task_id, STATUS_PENDING, now, now, "") for task_id in task_ids]
    with _get_conn(path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO tasks (task_id, status, created_at, updated_at, claimed_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=row["id"],
        task_id=row["task_id"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        claimed_by=row["claimed_by"],
    )


def claim_next_task(path: Path, consumer_id: str) -> Optional[Task]:
    init_db(path)
    with _get_conn(path) as conn:
        conn.isolation_level = None  # manual transactions
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (STATUS_PENDING,),
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        now = _utc_now_iso()
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, claimed_by = ?, updated_at = ?
            WHERE id = ?
            """,
            (STATUS_IN_PROGRESS, consumer_id, now, row["id"]),
        )
        conn.execute("COMMIT")
        return Task(
            id=row["id"],
            task_id=row["task_id"],
            status=STATUS_IN_PROGRESS,
            created_at=row["created_at"],
            updated_at=now,
            claimed_by=consumer_id,
        )


def mark_done(path: Path, task_id: str) -> bool:
    init_db(path)
    with _get_conn(path) as conn:
        now = _utc_now_iso()
        cur = conn.execute(
            """
            UPDATE tasks
            SET status = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (STATUS_DONE, now, task_id),
        )
        conn.commit()
        return cur.rowcount > 0
