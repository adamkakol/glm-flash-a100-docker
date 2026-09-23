"""Thread-safe, low-frequency progress snapshots; no third-party dependencies."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import threading
import time

try:
    from .deployment import atomic_write
except ImportError:
    from deployment import atomic_write


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class BenchmarkProgress:
    """Collect stream activity in memory; publish at most once per interval."""
    def __init__(self, path=None, interval=2):
        self.path = Path(path) if path else None
        self.interval = interval
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.publish_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = {'stage': 'Checking server', 'status': 'running', 'requests': {}}

    def stage(self, name):
        with self.lock:
            self.state['stage'] = name

    def request(self, number, **fields):
        with self.lock:
            self.state['requests'].setdefault(str(number), {}).update(fields)

    def output(self, number):
        with self.lock:
            request = self.state['requests'].setdefault(str(number), {})
            request.update(status='generating', last_output_elapsed_s=time.monotonic() - self.started)
            request['output_events'] = request.get('output_events', 0) + 1

    def finish(self, status):
        with self.lock:
            self.state['status'] = status
        self.publish()

    def snapshot(self):
        with self.lock:
            return {**deepcopy(self.state), 'elapsed_s': time.monotonic() - self.started,
                    'updated_at': datetime.now(timezone.utc).isoformat(), 'updated_unix_s': time.time()}

    def publish(self):
        if self.path:
            # Keep file I/O outside the lock touched on every output event.
            with self.publish_lock:
                atomic_write(self.path, json.dumps(self.snapshot(), indent=2) + '\n')

    def worker(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.publish()
            except OSError:
                # The authoritative final report still reports file failures in the main thread.
                pass

    def __enter__(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.publish()
            self.thread = threading.Thread(target=self.worker, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if exc_type:
            with self.lock:
                self.state['status'] = 'interrupted' if issubclass(exc_type, KeyboardInterrupt) else 'failed'
        self.publish()


class TuningProgress:
    """Newline-based display works in terminals, SSH, redirected logs and tmux."""
    def __init__(self, directory, total, interval=15):
        self.directory = Path(directory)
        self.total, self.interval = total, interval
        self.started = self.stage_started = self.candidate_started = time.monotonic()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.successful_durations = []
        self.benchmark_path = None
        self.console = True
        self.state = {'status': 'running', 'phase': 'setup', 'stage': 'Preparing tuner',
                      'candidate': None, 'candidate_index': 0, 'matrix_completed': 0,
                      'matrix_total': total, 'last_metrics': None}

    def set(self, *, phase=None, stage=None, candidate=None, index=None, status=None, metrics=None, message=None):
        with self.lock:
            if phase is not None:
                self.state['phase'] = phase
            if stage is not None:
                self.state['stage'] = stage
                self.stage_started = time.monotonic()
                self.benchmark_path = None
            if candidate is not None:
                self.state['candidate'] = candidate
            if index is not None:
                self.state['candidate_index'] = index
                self.candidate_started = time.monotonic()
            if status is not None:
                self.state['status'] = status
            if metrics is not None:
                self.state['last_metrics'] = deepcopy(metrics)
                self.state['last_metrics_for'] = {k: self.state[k] for k in ('phase', 'candidate', 'stage')}
            self.emit(message)

    def follow(self, path):
        with self.lock:
            self.benchmark_path = Path(path)

    def complete_candidate(self, passed, elapsed):
        with self.lock:
            self.state['matrix_completed'] += 1
            if passed:
                self.successful_durations.append(elapsed)
            self.state['stage'] = 'Candidate passed' if passed else 'Candidate rejected'
            self.stage_started = time.monotonic()
            self.benchmark_path = None
            self.emit()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            value = {**deepcopy(self.state), 'elapsed_s': now - self.started,
                     'stage_elapsed_s': now - self.stage_started,
                     'updated_at': datetime.now(timezone.utc).isoformat(),
                     'matrix_eta_s': None, 'eta_scope': 'remaining matrix only; excludes boundary, expansion and final checks'}
            remaining = max(0, self.total - self.state['matrix_completed'])
            if self.state['phase'] == 'matrix' and len(self.successful_durations) >= 2 and remaining:
                active_elapsed = (now - self.candidate_started
                                  if self.state['candidate_index'] > self.state['matrix_completed'] else 0)
                value['matrix_eta_s'] = max(0, statistics.median(self.successful_durations) * remaining - active_elapsed)
            if self.benchmark_path:
                try:
                    value['benchmark'] = json.loads(self.benchmark_path.read_text())
                except (OSError, ValueError):
                    value['benchmark'] = None
            return value

    def render(self, value, message=None):
        complete = value['matrix_completed']
        filled = int(20 * complete / max(1, self.total))
        bar = '#' * filled + '-' * (20 - filled)
        prefix = (f'[{duration(value["elapsed_s"])}] Matrix [{bar}] '
                  f'{complete}/{self.total} ({100 * complete / max(1, self.total):.0f}%)')
        if value['phase'] == 'matrix':
            prefix += f' | candidate {value["candidate_index"]}/{self.total}'
        else:
            prefix += f' | {value["phase"]}'
        prefix += f' | {value["stage"]} ({duration(value["stage_elapsed_s"])} in stage)'
        if value['matrix_eta_s'] is not None:
            prefix += f' | matrix ETA ~{duration(value["matrix_eta_s"])}'
        lines = [prefix]
        if value['candidate']:
            lines.append('  ' + value['candidate'])
        benchmark = value.get('benchmark')
        if benchmark:
            age = max(0, time.time() - benchmark['updated_unix_s'])
            lines.append(f'  {benchmark["stage"]}; snapshot age {age:.0f}s')
            for number, request in sorted(benchmark['requests'].items(), key=lambda item: int(item[0])):
                description = f'  Request {number}: {request.get("status", "preparing")}'
                if 'input_tokens' in request:
                    description += f'; input {request["input_tokens"]:,} tokens'
                if 'output_events' in request:
                    last = benchmark['elapsed_s'] - request['last_output_elapsed_s'] + age
                    description += f'; {request["output_events"]:,} output events; last output {last:.0f}s ago'
                if 'completion_tokens' in request:
                    description += f'; completed {request["completion_tokens"]:,} tokens'
                lines.append(description)
        if message:
            lines.append('  ' + message)
        return '\n'.join(lines)

    def emit(self, message=None):
        with self.lock:
            value = self.snapshot()
            text = self.render(value, message)
            # Reporting must not prevent failure recovery if its destination becomes unwritable.
            try:
                atomic_write(self.directory / 'progress.json', json.dumps(value, indent=2) + '\n')
                with (self.directory / 'progress.log').open('a') as log:
                    log.write(text + '\n')
            except OSError as exc:
                text += f'\n  Progress file write failed: {exc}'
            if self.console:
                try:
                    print(text, flush=True)
                except OSError:
                    self.console = False

    def worker(self):
        while not self.stop_event.wait(self.interval):
            self.emit()

    def __enter__(self):
        self.emit(f'Live status: {self.directory / "progress.json"}')
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.emit()
