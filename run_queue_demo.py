from __future__ import annotations

import subprocess
import sys
from pathlib import Path

QUEUE_PATH = Path(__file__).resolve().parent / "queue.csv"


def main() -> None:
    base_cmd = [sys.executable]

    # Seed 100 tasks
    print("Seeding 100 tasks...")
    subprocess.run(base_cmd + ["producer.py", "--count", "100", "--queue", str(QUEUE_PATH)], check=True)

    # Launch 4 consumers in the foreground with exit_when_empty to finish once the queue is drained.
    consumer_cmd = base_cmd + [
        "consumer.py",
        "--queue",
        str(QUEUE_PATH),
        "--exit-when-empty",
    ]
    consumers = []
    for i in range(4):
        name = f"consumer-{i+1}"
        cmd = consumer_cmd + ["--name", name]
        proc = subprocess.Popen(cmd)
        consumers.append(proc)
        print(f"Started {name} with PID {proc.pid}")

    # Wait for consumers to finish.
    for proc in consumers:
        proc.wait()
    print("All consumers finished. Queue processing complete.")


if __name__ == "__main__":
    main()
