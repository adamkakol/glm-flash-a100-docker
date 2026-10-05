"""Durable, private reports, exclusive operations, and live progress."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def fingerprint(config: dict, source_dir: Path) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in sorted(source_dir.rglob("*.py")):
        if "tests" not in path.parts:
            digest.update(str(path.relative_to(source_dir)).encode())
            digest.update(path.read_bytes())
    for path in sorted((source_dir / "fixtures").glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@contextlib.contextmanager
def exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another deployment operation holds " + str(path)) from exc
        yield
    finally:
        os.close(fd)


class Progress:
    def __init__(self, directory: Path, interval: float = 15):
        self.directory = directory
        self.interval = interval
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.started = time.monotonic()
        self.data = {"status": "running", "stage": "preflight", "completed": 0,
                     "total": 0, "activity": {}, "updated_at": time.time()}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        self.update()
        return self

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.data["elapsed_s"] = round(time.monotonic() - self.started, 1)
            self.data["updated_at"] = time.time()
            atomic_json(self.directory / "progress.json", self.data)

    def event(self, *args, **kwargs):
        # The client may report a dictionary or a named event with fields.
        activity = args[0] if args and isinstance(args[0], dict) else {"event": str(args[0]) if args else "request", **kwargs}
        with self.lock:
            if "request" in activity:
                self.data.setdefault("requests", {})[str(activity["request"])] = activity
            self.update(activity=activity)

    def log(self, message: str):
        with self.lock:
            line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
            print(line, flush=True)
            with (self.directory / "progress.log").open("a") as stream:
                stream.write(line + "\n")

    def _heartbeat(self):
        while not self.done.wait(self.interval):
            with self.lock:
                self.update()
                n, total = self.data["completed"], self.data["total"]
                fill = int(20 * n / total) if total else 0
                bar = "#" * fill + "." * (20 - fill)
                activity = self.data.get("activity", {})
                self.log(f"[{bar}] {self.data['stage']} {n}/{total} "
                         f"candidate={self.data.get('candidate', '-')} "
                         f"elapsed={self.data['elapsed_s']:.0f}s "
                         f"activity={activity.get('stage', activity.get('event', '-'))} "
                         f"request={activity.get('request', '-')}")

    def __exit__(self, kind, value, tb):
        self.done.set()
        self.thread.join(timeout=2)
        self.update(status="interrupted" if kind else self.data.get("status", "completed"))
