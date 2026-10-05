"""Observe complete workloads; topology alone never proves NVLink traffic."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.request

_COUNTER_TERMS = ("preemption", "spec_decode_num_accepted", "spec_decode_num_draft", "prefix_cache_hits", "prefix_cache_queries")
_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def metrics(text: str) -> dict[str, float]:
    result = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r'([^\s{]+)(?:\{[^}]*\})?\s+([-+.\deE]+)(?:\s|$)', line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                result[match[1]] = result.get(match[1], 0) + value
    return result


def read_metrics(url: str, key: str, timeout=5) -> dict:
    request = urllib.request.Request(url.rstrip("/") + "/metrics",
                                     headers={"Authorization": "Bearer " + key})
    deadline = time.monotonic() + timeout
    with urllib.request.urlopen(request, timeout=timeout) as response:
        sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
        def abort():
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        timer = threading.Timer(max(.001, deadline - time.monotonic()), abort)
        timer.daemon = True
        timer.start()
        try:
            payload = response.read(8 * 1024 * 1024 + 1)
            if time.monotonic() > deadline:
                raise TimeoutError("metrics request exceeded deadline")
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError("metrics response exceeded size bound")
            return metrics(payload.decode())
        finally:
            timer.cancel()


def wait_idle(backends: list[str], key: str, timeout=30) -> dict:
    """Require every direct backend to reclaim cancelled running/queued work."""
    if not backends or len(set(backends)) != len(backends) or timeout <= 0:
        raise ValueError("unique backends and a positive timeout are required")
    started, latest, errors, attempts = time.monotonic(), {}, {}, 0
    deadline = started + timeout
    while time.monotonic() < deadline:
        attempts += 1
        latest, errors = {}, {}
        for url in backends:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors[url] = "deadline exceeded"
                break
            try:
                values = read_metrics(url, key, timeout=min(5, remaining))
                gauges = {name: values.get(name) for name in _GAUGES}
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in gauges.values()):
                    raise ValueError("missing or invalid scheduler gauges")
                latest[url] = gauges
            except Exception as exc:
                errors[url] = type(exc).__name__
        if not errors and len(latest) == len(backends) and all(v == 0 for gauges in latest.values() for v in gauges.values()):
            return {"passed": True, "duration_s": time.monotonic() - started, "attempts": attempts, "backends": latest}
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(.25, remaining))
    return {"passed": False, "duration_s": time.monotonic() - started, "attempts": attempts,
            "backends": latest, "errors": errors, "reason": "not every backend returned to zero running and waiting requests"}


class Telemetry:
    def __init__(self, directory: Path, backends: list[str], key: str, uuids: list[str], interval=2, max_sample_gap_s=15):
        if not backends or not uuids or interval <= 0 or max_sample_gap_s <= interval:
            raise ValueError("backends, GPUs and a sampling gap larger than the positive interval are required")
        self.directory, self.backends, self.key = directory, backends, key
        self.uuids, self.interval, self.max_sample_gap_s = set(uuids), interval, max_sample_gap_s
        self.finished = threading.Event()
        self.samples = 0
        self.minimum_free_mib = {}
        self.errors = set()
        self.before, self.after, self.previous_metrics = {}, {}, {}
        self.nvlink_samples = 0
        self.sample_times = []
        self.started = self.ended = None
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _metrics(self):
        values = {}
        for url in self.backends:
            try:
                current = read_metrics(url, self.key)
                if any(not math.isfinite(value) for value in current.values()):
                    raise ValueError("non-finite metric")
                if not any("preemption" in name for name in current):
                    raise ValueError("preemption counter missing")
                previous = self.previous_metrics.get(url)
                if previous is not None:
                    tracked = {name for name in set(previous) | set(current) if any(term in name for term in _COUNTER_TERMS)}
                    for name in tracked:
                        if name not in previous or name not in current:
                            self.errors.add("counter appeared or disappeared: " + name)
                        elif current[name] < previous[name] or current[name] < 0:
                            self.errors.add("counter reset or invalid: " + name)
                elif any(value < 0 for name, value in current.items() if any(term in name for term in _COUNTER_TERMS)):
                    raise ValueError("negative counter")
                values[url] = current
                self.previous_metrics[url] = current
            except Exception as exc:
                self.errors.add("metrics: " + type(exc).__name__)
        return values

    def __enter__(self):
        self.before = self._metrics()
        self.started = time.monotonic()
        # Guarantee a baseline GPU sample even for a very short workload.
        with (self.directory / "telemetry.jsonl").open("a") as stream:
            self._sample(stream)
        self.thread.start()
        return self

    def _sample(self, stream):
        sample = {"time": time.time(), "metrics": self._metrics()}
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            gpus = []
            for line in result.stdout.splitlines():
                cells = [x.strip() for x in line.split(",")]
                if len(cells) == 5 and cells[0] in self.uuids:
                    free = float(cells[1])
                    if not math.isfinite(free) or free < 0:
                        raise ValueError("invalid GPU memory sample")
                    gpus.append({"uuid": cells[0], "free_mib": free, "utilization": cells[2],
                                 "temperature_c": cells[3], "power_w": cells[4]})
            if len(gpus) != len(self.uuids) or {gpu["uuid"] for gpu in gpus} != self.uuids:
                raise ValueError("GPU sample does not cover every selected UUID exactly once")
            for gpu in gpus:
                uuid, free = gpu["uuid"], gpu["free_mib"]
                self.minimum_free_mib[uuid] = min(self.minimum_free_mib.get(uuid, free), free)
            sample["gpus"] = gpus
            self.samples += 1
            self.sample_times.append(time.monotonic())
        except Exception as exc:
            self.errors.add("GPU sample: " + type(exc).__name__)
        if self.samples % 5 == 1:
            try:
                output = subprocess.run(["nvidia-smi", "nvlink", "--getthroughput", "d"],
                                        capture_output=True, text=True, timeout=5)
                sample["nvlink"] = {"returncode": output.returncode, "output": output.stdout[:16384], "error": output.stderr[:2048]}
                self.nvlink_samples += output.returncode == 0
            except Exception as exc:
                sample["nvlink"] = {"error": type(exc).__name__}
        stream.write(json.dumps(sample, allow_nan=False) + "\n")
        stream.flush()

    def _loop(self):
        try:
            with (self.directory / "telemetry.jsonl").open("a") as stream:
                while not self.finished.wait(self.interval):
                    self._sample(stream)
        except Exception as exc:
            self.errors.add("sampler stopped: " + type(exc).__name__)

    def __exit__(self, *args):
        self.finished.set()
        self.thread.join(timeout=max(15, 5 * (len(self.backends) + 2) + 1))
        if self.thread.is_alive():
            self.errors.add("sampler thread failed to stop")
        self.ended = time.monotonic()
        self.after = self._metrics()

    def summary(self, minimum_free_gib=3) -> dict:
        deltas, counters_valid = {}, True
        for url in self.backends:
            start, end = self.before.get(url, {}), self.after.get(url, {})
            tracked = {name for name in set(start) | set(end) if any(term in name for term in _COUNTER_TERMS)}
            for name in tracked:
                if name not in start or name not in end or end[name] < start[name] or start[name] < 0:
                    counters_valid = False
                    continue
                deltas[name] = deltas.get(name, 0) + end[name] - start[name]
        preemptions = sum(v for k, v in deltas.items() if "preemption" in k)
        enough_samples = bool(self.samples) and self.uuids == self.minimum_free_mib.keys()
        complete_metrics = len(self.before) == len(self.backends) == len(self.after)
        preemption_observed = all(any("preemption" in name for name in values.get(url, {}))
                                  for values in (self.before, self.after) for url in self.backends)
        timeline = [self.started or time.monotonic(), *self.sample_times, self.ended or time.monotonic()]
        gap = max((b - a for a, b in zip(timeline, timeline[1:])), default=math.inf)
        stopped = self.ended is not None and not self.thread.is_alive()
        passed = (enough_samples and complete_metrics and preemption_observed and counters_valid and stopped
                  and not self.errors and gap <= self.max_sample_gap_s and preemptions == 0
                  and min(self.minimum_free_mib.values(), default=0) >= minimum_free_gib * 1024)
        return {"passed": bool(passed), "samples": self.samples, "minimum_free_mib": self.minimum_free_mib,
                "metrics_complete": complete_metrics, "preemption_counter_present": preemption_observed,
                "counters_monotonic": counters_valid and not any("counter " in e for e in self.errors),
                "sampler_stopped": stopped, "max_sample_gap_s": gap, "allowed_sample_gap_s": self.max_sample_gap_s,
                "counter_deltas": deltas, "preemptions": preemptions, "nvlink_counter_samples": self.nvlink_samples,
                "nvlink_traffic_verified": False,
                "nvlink_note": "Inspect timestamped counters in telemetry.jsonl; topology and counter availability alone are not traffic proof.",
                "errors": sorted(self.errors)}
