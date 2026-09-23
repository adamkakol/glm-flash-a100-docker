#!/usr/bin/env python3
"""Offline maintenance benchmark: select the fastest eligible profile on this host."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

from scripts.deployment import Profile, atomic_write, read_env, write_profile
from scripts.progress import TuningProgress, duration

ROOT = Path(__file__).resolve().parent
WEIGHTS = {'ttft': .25, 'decode_cost': .45, 'extra_ttft': .20, 'pause': .10}


def run(command, *, root=ROOT, timeout=120):
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {(result.stderr or result.stdout)[-1500:]}')
    return result.stdout.strip()


def compose(*args, root=ROOT, timeout=120):
    return run(['docker', 'compose', *args], root=root, timeout=timeout)


def run_logged(command, *, root, log, timeout):
    """Persist child output and ensure interruption/timeout cannot leave a benchmark running."""
    with log.open('w') as output:
        process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise


def metrics_text(report):
    metrics = report.get('metrics', {})
    if not metrics:
        return 'PASS'
    text = (f'PASS | worst TTFT {metrics["worst_ttft_s"]:.1f}s | '
            f'slower session {metrics["worst_decode_tokens_per_s"]:.2f} tokens/s | '
            f'longest output pause {metrics["worst_max_stream_gap_s"]:.2f}s | '
            f'overlap {100 * metrics["generation_overlap_fraction"]:.0f}%')
    if 'extra_user_worst_ttft_s' in metrics:
        text += f' | extra-session TTFT {metrics["extra_user_worst_ttft_s"]:.1f}s'
    return text


def costs(record):
    trials = record['trials']
    return {
        'ttft': statistics.median(t['long']['metrics']['worst_ttft_s'] for t in trials),
        'decode_cost': 1 / max(.000001, statistics.median(
            t['long']['metrics']['worst_decode_tokens_per_s'] for t in trials)),
        'extra_ttft': statistics.median(t['mixed']['metrics']['extra_user_worst_ttft_s'] for t in trials),
        'pause': max(.001, statistics.median(t['mixed']['metrics']['worst_max_stream_gap_s'] for t in trials)),
    }


def rank_profiles(records):
    eligible = [r for r in records if r.get('status') == 'passed']
    if not eligible:
        return []
    values = [costs(r) for r in eligible]
    best = {key: max(.000001, min(v[key] for v in values)) for key in WEIGHTS}
    for record, value in zip(eligible, values):
        record['costs'] = value
        record['score'] = math.exp(sum(WEIGHTS[k] * math.log(max(.000001, value[k]) / best[k])
                                       for k in WEIGHTS))
    return sorted(eligible, key=lambda r: r['score'])


def no_material_regression(candidate, baseline, tolerance):
    a, b = costs(candidate), costs(baseline)
    return all(a[k] <= max(.001, b[k]) * (1 + tolerance) for k in WEIGHTS)


class GPUWatch:
    """Sample only the selected GPUs. Missing measurements make a trial ineligible."""
    def __init__(self, uuids):
        self.uuids = set(uuids)
        self.stop_event = threading.Event()
        self.samples = []
        self.errors = []
        self.thread = threading.Thread(target=self.worker, daemon=True)

    def worker(self):
        while not self.stop_event.is_set():
            try:
                text = run(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], timeout=10)
                selected = {}
                for line in text.splitlines():
                    uuid, free, used, total = [s.strip() for s in line.split(',')]
                    if uuid in self.uuids:
                        selected[uuid] = {'free_mib': int(free), 'used_mib': int(used), 'total_mib': int(total)}
                if set(selected) != self.uuids:
                    raise RuntimeError('Selected GPU missing from nvidia-smi')
                self.samples.append(selected)
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop_event.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join(timeout=12)

    def summary(self):
        if self.errors or not self.samples:
            raise RuntimeError('GPU measurement failed: ' + '; '.join(self.errors))
        return {uuid: {'minimum_free_mib': min(s[uuid]['free_mib'] for s in self.samples),
                       'peak_used_mib': max(s[uuid]['used_mib'] for s in self.samples)}
                for uuid in sorted(self.uuids)}


class Tuner:
    def __init__(self, args, root=ROOT):
        self.args, self.root = args, root
        self.env = read_env(root)
        self.uuids = [self.env[f'GPU{i}_UUID'] for i in range(3)]
        self.directory = root / 'reports' / ('autotune-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
        self.directory.mkdir(parents=True)
        self.originals = {name: (root / name).read_bytes() if (root / name).exists() else None
                          for name in ('config.yml', 'deployment.json')}
        for name, data in self.originals.items():
            if data is not None:
                (self.directory / ('original-' + name)).write_bytes(data)
        cid = compose('ps', '-q', 'server', root=root)
        self.was_running = bool(cid and run(['docker', 'inspect', '--format', '{{.State.Running}}', cid]) == 'true')
        config = json.loads(compose('config', '--format', 'json', root=root))
        image = config['services']['server']['image']
        image_id = run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'])
        self.report = {'status': 'running', 'objective_weights': WEIGHTS, 'records': [],
                       'image_id': image_id, 'gpu_uuids': self.uuids,
                       'versions': json.loads((root / 'versions.json').read_text()),
                       'host': run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv']),
                       'topology': run(['nvidia-smi', 'topo', '-m']),
                       'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
        total = len(args.modes) * len(args.chunks) * len(args.draft_tokens)
        self.progress = TuningProgress(self.directory, total, args.progress_interval)
        self.save()

    def save(self):
        atomic_write(self.directory / 'summary.json', json.dumps(self.report, indent=2) + '\n')

    def stop(self):
        self.progress.set(stage='Stopping server')
        compose('stop', '-t', '30', 'server', root=self.root, timeout=60)

    def start(self, profile):
        self.stop()
        write_profile(self.root, profile, selected_by='autotune-candidate')
        self.progress.set(stage='Starting container', candidate=profile.name)
        compose('up', '-d', '--no-build', '--pull', 'never', '--force-recreate', 'server', root=self.root)
        self.wait_healthy()

    def wait_healthy(self):
        self.progress.set(stage='Loading model and warming kernels; waiting for healthy status')
        deadline = time.monotonic() + self.args.startup_timeout
        cid = compose('ps', '-aq', 'server', root=self.root)
        if not cid:
            raise RuntimeError('Server container not created')
        while time.monotonic() < deadline:
            state = json.loads(run(['docker', 'inspect', '--format', '{{json .State}}', cid]))
            if state.get('Restarting') or not state.get('Running'):
                raise RuntimeError('Candidate exited or restarted during loading; inspect server logs locally')
            status = state.get('Health', {}).get('Status')
            if status == 'healthy':
                self.progress.emit('Server healthy')
                return
            if status == 'unhealthy':
                raise RuntimeError('Candidate healthcheck failed')
            time.sleep(2)
        raise RuntimeError('Candidate startup timeout')

    def check(self, profile, kind, trial, *, tokens=None, output=None, users=None):
        path = self.directory / f'{profile.name}-{trial}-{kind}.json'
        label = f'{kind} test {trial+1}/{self.args.repeats}' if isinstance(trial, int) else f'{kind}: {trial}'
        self.progress.set(stage=label)
        activity = self.directory / 'benchmark-progress.json'
        activity.unlink(missing_ok=True)
        self.progress.follow(activity)
        command = [sys.executable, 'scripts/api_check.py', kind, '--tokens', str(tokens or self.args.tokens),
                   '--max-output', str(output or self.args.output_tokens), '--users', str(users or (profile.max_batch_size if kind == 'mixed' else 2)),
                   '--timeout', str(self.args.request_timeout), '--report', str(path),
                   '--reasoning-effort', self.args.reasoning_effort, '--image-size', str(self.args.image_size),
                   '--progress-json', str(activity)]
        if kind == 'mixed' and profile.vision:
            command.append('--with-images')
        if self.args.corpus:
            command.extend(['--corpus', str(self.args.corpus.resolve())])
        # Includes a separate allowance for prompt construction/tokenization.
        log = path.with_suffix('.log')
        code = run_logged(command, root=self.root, log=log, timeout=self.args.request_timeout + 900)
        if not path.exists():
            raise RuntimeError(f'Benchmark produced no report; see {log}')
        report = json.loads(path.read_text())
        if code or report.get('status') != 'passed':
            raise RuntimeError(f'{kind} check failed: {report.get("errors", [])}; details: {log}')
        self.progress.set(metrics=report.get('metrics', {}), message=metrics_text(report))
        report['report_file'] = str(path.relative_to(self.root))
        return report

    def evaluate(self, profile):
        started = time.monotonic()
        record = {'profile': asdict(profile), 'name': profile.name, 'status': 'running', 'trials': []}
        self.report['records'].append(record)
        self.progress.set(stage='Preparing candidate', candidate=profile.name)
        self.save()
        try:
            # Watch loading too: reserve must survive all measured stages.
            self.stop()
            with GPUWatch(self.uuids) as watch:
                self.start(profile)
                self.check(profile, 'smoke', 'warmup')
                if profile.vision:
                    vision = self.check(profile, 'vision', 'warmup')
                    record['vision_report'] = vision['report_file']
                for trial in range(self.args.repeats):
                    long = self.check(profile, 'long', trial)
                    mixed = self.check(profile, 'mixed', trial)
                    record['trials'].append({kind: {'metrics': value['metrics'], 'report_file': value['report_file']}
                                             for kind, value in (('long', long), ('mixed', mixed))})
                    self.save()
            record['gpu_memory'] = watch.summary()
            if any(v['minimum_free_mib'] < self.args.min_free_mib for v in record['gpu_memory'].values()):
                raise RuntimeError('Insufficient measured per-GPU VRAM headroom')
            record['status'] = 'passed'
        except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as exc:
            record['status'] = 'failed'
            record['error'] = str(exc)
            self.progress.emit(f'Rejected: {exc}')
        finally:
            try:
                self.stop()
            finally:
                record['elapsed_s'] = time.monotonic() - started
                if record['status'] == 'running':
                    record['status'] = 'interrupted'
                self.save()
        if record['status'] == 'passed':
            free = min(v['minimum_free_mib'] for v in record['gpu_memory'].values()) / 1024
            self.progress.emit(f'Candidate passed in {duration(record["elapsed_s"])}; minimum sampled free VRAM {free:.2f} GiB')
        return record

    def boundary(self, profile):
        self.progress.set(phase='context boundary', stage=f'Checking {profile.max_seq_len:,}-token context', candidate=profile.name)
        # Exercise the advertised ceiling, not just the two-user target. This is a capacity check.
        self.stop()
        with GPUWatch(self.uuids) as watch:
            self.start(profile)
            report = self.check(profile, 'long', 'context-boundary', users=1,
                                tokens=profile.max_seq_len - self.args.output_tokens - 2304)
        memory = watch.summary()
        if any(v['minimum_free_mib'] < self.args.min_free_mib for v in memory.values()):
            raise RuntimeError('Context-boundary run violates VRAM headroom')
        return {'report_file': report['report_file'], 'gpu_memory': memory}

    def restore(self):
        self.progress.set(phase='restoring', stage='Restoring original configuration and service state')
        self.stop()
        for name, data in self.originals.items():
            path = self.root / name
            if data is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, data.decode())
        if self.was_running:
            compose('up', '-d', '--no-build', '--pull', 'never', '--force-recreate', 'server', root=self.root)
            self.wait_healthy()

    def optimize(self, base):
        success = False
        try:
            matrix = itertools.product(self.args.modes, self.args.chunks, self.args.draft_tokens)
            for index, (mode, chunk, draft) in enumerate(matrix, 1):
                profile = replace(base, mode=mode, chunk_size=chunk, draft_tokens=draft).validate()
                self.progress.set(phase='matrix', stage='Beginning candidate', candidate=profile.name, index=index)
                record = self.evaluate(profile)
                self.progress.complete_candidate(record['status'] == 'passed', record['elapsed_s'])
            self.progress.set(phase='ranking', stage='Ranking eligible configurations')
            ranked = rank_profiles(self.report['records'])
            winner = None
            for candidate in ranked:
                profile = Profile(**candidate['profile'])
                try:
                    candidate['context_boundary'] = self.boundary(profile)
                    winner = candidate
                    break
                except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as exc:
                    candidate['status'] = 'failed'; candidate['boundary_error'] = str(exc)
                    self.progress.emit(f'Context check failed; trying next candidate: {exc}')
                    self.stop(); self.save()
            if winner is None:
                raise RuntimeError('No configuration passed the capacity, concurrency and context-boundary checks')
            # Larger context/cache only wins if the target workload remains efficient.
            if not self.args.no_expand and base.max_seq_len < 1048576:
                expanded = replace(Profile(**winner['profile']), max_seq_len=1048576,
                                   cache_size=max(base.cache_size, 1572864))
                self.progress.set(phase='context expansion', stage='Testing larger cache/context', candidate=expanded.name)
                larger = self.evaluate(expanded)
                if larger['status'] == 'passed' and no_material_regression(larger, winner, self.args.expansion_tolerance):
                    try:
                        larger['context_boundary'] = self.boundary(expanded)
                        winner = larger
                    except (RuntimeError, subprocess.SubprocessError, OSError, ValueError) as exc:
                        larger['status'] = 'failed'; larger['boundary_error'] = str(exc)
                        self.progress.emit(f'Expanded context rejected: {exc}')
                else:
                    larger['selected'] = False
                    larger['selection_note'] = 'Capacity failure or material target-workload latency/throughput regression'
                    self.progress.emit('Keeping smaller context: expansion failed or regressed')
            selected = Profile(**winner['profile'])
            self.progress.set(phase='final validation', stage='Starting selected configuration', candidate=selected.name)
            self.start(selected)
            self.check(selected, 'smoke', 'selected')
            if selected.vision:
                self.check(selected, 'vision', 'selected')
            evidence = str(self.directory.relative_to(self.root) / 'summary.json')
            write_profile(self.root, selected, selected_by='measured-autotune', evidence=evidence)
            self.report['status'] = 'selected'; self.report['selected'] = asdict(selected)
            self.report['selection_scope'] = 'Best measured eligible candidate for this workload; not a universal optimum'
            self.save(); success = True
            self.progress.set(phase='complete', stage='Selected configuration is running', status='selected')
            self.progress.emit(f'Selected {selected.name}; evidence: {evidence}')
        finally:
            if not success:
                self.report['status'] = 'restoring-original'
                self.save()
                try:
                    self.restore()
                    self.report['status'] = 'failed-original-restored'
                    self.progress.set(phase='stopped', stage='Run did not complete; original configuration restored', status='failed-original-restored')
                except Exception as exc:
                    self.report['status'] = 'failed-restore-needs-attention'
                    self.report['restore_error'] = str(exc)
                    self.progress.set(phase='failed', stage='Restoration needs attention', status='failed-restore-needs-attention')
                    self.progress.emit(str(exc))
                    raise
                finally:
                    self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=['nccl', 'native', 'layer'], default=['nccl', 'native', 'layer'])
    parser.add_argument('--chunks', nargs='+', type=int, choices=[1024, 2048, 4096], default=[2048, 4096])
    parser.add_argument('--draft-tokens', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=260000)
    parser.add_argument('--output-tokens', type=int, default=4096)
    parser.add_argument('--min-free-mib', type=int, default=3072)
    parser.add_argument('--startup-timeout', type=int, default=1800)
    parser.add_argument('--request-timeout', type=int, default=7200)
    parser.add_argument('--no-expand', action='store_true')
    parser.add_argument('--expansion-tolerance', type=float, default=.05)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='max')
    parser.add_argument('--image-size', type=int, default=1024, help='Square image side for vision probes and mixed arrivals')
    parser.add_argument('--progress-interval', type=int, default=15, help='Seconds between live terminal/status updates (5–300)')
    parser.add_argument('--plan', action='store_true', help='Print the experiment matrix; do not touch Docker or config')
    args = parser.parse_args()
    if not 5 <= args.progress_interval <= 300:
        parser.error('progress-interval must be 5–300 seconds')
    if not 28 <= args.image_size <= 4096:
        parser.error('image-size must be between 28 and 4096 pixels')
    if args.repeats < 1 or args.tokens < 260000 or args.output_tokens < 4096:
        parser.error('Acceptance tuning requires >=1 repeat, >=260000 input tokens and >=4096 output tokens')
    if args.min_free_mib < 2048 or min(args.startup_timeout, args.request_timeout) < 30:
        parser.error('Retain >=2048 MiB measured VRAM headroom and meaningful timeouts')
    if not 0 <= args.expansion_tolerance <= .2:
        parser.error('expansion-tolerance must be 0–0.2')
    if args.plan:
        print(json.dumps({'candidates': list(itertools.product(args.modes, args.chunks, args.draft_tokens)),
                          'repeats': args.repeats, 'input_per_long_session': args.tokens,
                          'output_per_long_session': args.output_tokens, 'weights': WEIGHTS,
                          'expand_context_if_efficient': not args.no_expand, 'vision_test_image_size': args.image_size}, indent=2))
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (ROOT / '.autotune.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads((ROOT / 'deployment.json').read_text())
        digest = hashlib.sha256((ROOT / 'config.yml').read_bytes()).hexdigest()
        if digest != record['config_sha256']:
            raise RuntimeError('config.yml was manually edited. Run configure.py --force with explicit settings first; backup your edits')
        base = Profile(**record['profile']).validate()
        print('Maintenance run: disconnect clients. The server will restart between candidates. No models/images are downloaded.', flush=True)
        tuner = Tuner(args)
        with tuner.progress:
            tuner.optimize(base)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Autotune stopped: {exc}', file=sys.stderr)
        raise SystemExit(1)
