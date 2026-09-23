#!/usr/bin/env python3
"""Capacity and latency checks. All token counts include the chat template."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.request
import uuid

try:
    from .deployment import MODEL_NAME, read_env, validate_limits
except ImportError:
    from deployment import MODEL_NAME, read_env, validate_limits

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, url, key, timeout=120, reasoning_effort="max"):
        self.url, self.key, self.timeout = url.rstrip('/'), key, timeout
        self.reasoning_effort = reasoning_effort
        self.prompt_token_offset = 0

    def open(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers={
            'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'HTTP {exc.code}: {exc.read(4096).decode(errors="replace")}') from exc

    def json(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def token_count(self, messages):
        return self.json('/v1/token/encode', {'text': messages,
            'template_vars': {'reasoning_effort': self.reasoning_effort, 'clear_thinking': True}})['length'] + self.prompt_token_offset


    def calibrate_prompt_count(self, model):
        # Tabby's encode endpoint omits the assistant generation prefix. Measure
        # that constant offset using the same one-user-message shape as the tests.
        offsets = []
        for text in ['Count calibration. Reply briefly.', 'Another calibration with a different ending: 42']:
            messages = [{'role': 'user', 'content': text}]
            encoded = self.token_count(messages) - self.prompt_token_offset
            result = self.json('/v1/chat/completions', {'model': model, 'messages': messages,
                'max_tokens': 8, 'temperature': 1.0, 'top_p': .95,
                'chat_template_kwargs': {'reasoning_effort': self.reasoning_effort, 'clear_thinking': True}})
            offsets.append(result['usage']['prompt_tokens'] - encoded)
        if offsets[0] != offsets[1] or abs(offsets[0]) > 64:
            raise RuntimeError(f'Inconsistent chat template token accounting: {offsets}')
        self.prompt_token_offset = offsets[0]
        return offsets[0]


def make_prompt(client, target, max_input, corpus=None):
    nonce = uuid.uuid4().hex
    rng = random.Random(nonce)
    # Distinct prefixes avoid cross-request KV reuse; new prompts on every trial avoid warm-cache bias.
    if corpus:
        records = (corpus + '\n') * max(1, (target * 12 // (len(corpus) + 1)) + 1)
    else:
        records = '\n'.join(f'Record {i}: {rng.getrandbits(128):032x}.' for i in range(target // 8 + 100))
    prefix = f'Request {nonce}. Study this material for a synthetic serving benchmark.\n'
    suffix = ('\nWrite a lengthy structured analysis with varied numbered observations, examples, '
              'limitations, and possible applications. Continue in detail without repeating yourself.')

    def messages(length):
        return [{'role': 'user', 'content': prefix + records[:length] + suffix}]

    low, high = 0, min(len(records), max(1, target * 3))
    count = client.token_count(messages(high))
    while count < target:
        if high == len(records):
            raise RuntimeError('Unable to construct sufficiently long prompt; use a larger corpus')
        low, high = high, min(len(records), high * 2)
        count = client.token_count(messages(high))
    best, best_count = messages(high), count
    for _ in range(24):
        if target <= best_count <= min(target + 128, max_input):
            return best, best_count
        mid = (low + high) // 2
        candidate = messages(mid)
        count = client.token_count(candidate)
        if count >= target:
            high, best, best_count = mid, candidate, count
        else:
            low = mid + 1
    if target <= best_count <= max_input:
        return best, best_count
    raise RuntimeError('Could not fit measured prompt into requested context')


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def stream_request(client, model, messages, max_tokens, min_tokens, gate, number,
                   first_event, wait_for, deadline, epoch):
    gate.wait(timeout=30)
    # In the mixed test, extra users arrive only after both long sessions start decoding.
    for event in wait_for:
        if not event.wait(max(0, deadline - time.monotonic())):
            raise RuntimeError('Timed out waiting for the long sessions to begin generating')
    start = time.monotonic()
    stamps, usage, finish, output, done = [], None, None, '', False
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens,
            'min_tokens': min_tokens, 'temperature': 1.0, 'top_p': 0.95,
            'stream': True, 'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'reasoning_effort': client.reasoning_effort, 'clear_thinking': True},
            'loop_detect_window': 0}
    with client.open('/v1/chat/completions', body) as response:
        for raw in response:
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError('Benchmark wall-clock deadline exceeded')
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(str(event['error']))
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = (delta.get('reasoning_content') or '') + (delta.get('content') or '')
                if text:
                    stamps.append(now)
                    first_event.set()
                    if len(output) < 160:
                        output += text[:160 - len(output)]
                finish = choice.get('finish_reason') or finish
    end = time.monotonic()
    if not done or not usage or not stamps:
        raise RuntimeError('Incomplete stream or missing token usage/output')
    if finish not in {'stop', 'length'}:
        raise RuntimeError(f'Unexpected finish reason {finish}')
    count = usage.get('completion_tokens', 0)
    if count < min_tokens:
        raise RuntimeError(f'Only {count} output tokens; at least {min_tokens} required')
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    duration = stamps[-1] - stamps[0]
    return {'request': number, 'started_s': start - epoch, 'first_output_s': stamps[0] - epoch,
            'last_output_s': stamps[-1] - epoch, 'finished_s': end - epoch,
            'ttft_s': stamps[0] - start, 'total_s': end - start,
            'decode_tokens_per_s': count / duration if duration > 0 else 0,
            'p95_stream_gap_s': percentile(gaps, .95), 'max_stream_gap_s': max(gaps, default=0),
            'stream_events': len(stamps), 'event_times_s': [round(t - epoch, 6) for t in stamps],
            'finish_reason': finish, 'usage': usage, 'output_excerpt': output}


def summarize(results, expected_counts, minimum_output, long_users=2, mixed=False):
    failures = []
    for result, expected, minimum in zip(results, expected_counts, minimum_output):
        reported = result['usage']['prompt_tokens']
        # Counts include the calibrated generation prefix; allow only a tiny boundary discrepancy.
        if abs(reported - expected) > 2:
            failures.append(f'Request {result["request"]}: input changed from {expected} to {reported}')
        if result['usage']['completion_tokens'] < minimum:
            failures.append(f'Request {result["request"]}: insufficient output')
        if result['stream_events'] < 2 or result['decode_tokens_per_s'] <= 0:
            failures.append(f'Request {result["request"]}: no measurable sustained stream')
    primary = results[:long_users]
    overlap = max(0, min(r['last_output_s'] for r in primary) - max(r['first_output_s'] for r in primary))
    shortest = min(r['last_output_s'] - r['first_output_s'] for r in primary)
    overlap_fraction = overlap / shortest if shortest > 0 else 0
    if len(primary) > 1 and overlap_fraction < .1:
        failures.append('Long sessions did not show sustained overlapping output (minimum 10% of shorter stream)')
    if mixed:
        for result in results[long_users:]:
            if not (result['first_output_s'] < min(r['last_output_s'] for r in primary)):
                failures.append(f'Request {result["request"]}: extra session did not generate while both long sessions were active')
    metrics = {'worst_ttft_s': max(r['ttft_s'] for r in primary),
               'worst_decode_tokens_per_s': min(r['decode_tokens_per_s'] for r in primary),
               'worst_p95_stream_gap_s': max(r['p95_stream_gap_s'] for r in primary),
               'worst_max_stream_gap_s': max(r['max_stream_gap_s'] for r in primary),
               'generation_overlap_s': overlap, 'generation_overlap_fraction': overlap_fraction,
               'aggregate_tokens_per_s': sum(r['usage']['completion_tokens'] for r in results) /
                    max(.001, max(r['finished_s'] for r in results) - min(r['started_s'] for r in results))}
    if mixed:
        metrics['extra_user_worst_ttft_s'] = max(r['ttft_s'] for r in results[long_users:])
    return metrics, failures


def run_check(args, root=ROOT):
    env = read_env(root)
    url = args.url or f'http://127.0.0.1:{env.get("API_PORT", "5000")}'
    keys = json.loads((root / 'secrets/api_tokens.yml').read_text())['api_key']
    keys = [keys] if isinstance(keys, str) else keys
    users = args.users or (4 if args.mode == 'mixed' else 2)
    clients = [Client(url, keys[i % len(keys)], timeout=args.timeout, reasoning_effort=args.reasoning_effort) for i in range(users)]
    if clients[0].json('/health').get('status') != 'healthy':
        raise RuntimeError('Backend reports unhealthy')
    model = clients[0].json('/v1/model')
    if model['id'] != MODEL_NAME:
        raise RuntimeError('Unexpected model identity')
    params = model['parameters']
    validate_limits(params)
    if args.mode == 'smoke':
        result = clients[0].json('/v1/chat/completions', {'model': model['id'],
            'messages': [{'role': 'user', 'content': 'Reply with READY.'}], 'max_tokens': 512,
            'temperature': 1.0, 'top_p': .95, 'reasoning_effort': 'low'})
        message = result['choices'][0]['message']
        if not (message.get('content') or message.get('reasoning_content')):
            raise RuntimeError('No generated text')
        return {'status': 'passed', 'test': 'smoke', 'parameters': params, 'result': result}
    if users > params['max_batch_size']:
        raise RuntimeError('This concurrency benchmark needs enough active slots; queued requests are a separate test')
    if args.mode == 'mixed' and users < 3:
        raise RuntimeError('Mixed test requires at least three users')
    offset = clients[0].calibrate_prompt_count(model['id'])
    for client in clients:
        client.prompt_token_offset = offset
    minimum = args.min_output if args.min_output is not None else args.max_output
    corpus = args.corpus.read_text() if args.corpus else None
    prompts, counts, maximums, minimums = [], [], [], []
    for i, client in enumerate(clients):
        extra = args.mode == 'mixed' and i >= 2
        target, output = (args.short_tokens, args.short_output) if extra else (args.tokens, args.max_output)
        max_input = params['max_seq_len'] - output - 2048
        if target > max_input:
            raise RuntimeError('Input plus output and checkpoint margin exceeds context')
        messages, count = make_prompt(client, target, max_input, corpus)
        prompts.append(messages); counts.append(count)
        maximums.append(output); minimums.append(output if extra else minimum)
        print(f'Prepared request {i+1}: {count:,} input tokens', flush=True)
    # Conservative initial allocation, including recurrent checkpoint rounding.
    allocation = sum(((c + o + 2047) // 2048) * 2048 for c, o in zip(counts, maximums))
    if allocation > params['cache_size']:
        raise RuntimeError('Benchmark requests exceed shared cache; reduce the test workload or raise the measured cache budget')
    epoch = time.monotonic(); deadline = epoch + args.timeout
    gate = threading.Barrier(users); first_events = [threading.Event() for _ in clients]
    errors, results = [], []
    with ThreadPoolExecutor(max_workers=users) as pool:
        futures = [pool.submit(stream_request, client, model['id'], prompts[i], maximums[i], minimums[i],
                   gate, i+1, first_events[i], first_events[:2] if args.mode == 'mixed' and i >= 2 else [],
                   deadline, epoch) for i, client in enumerate(clients)]
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f'Request {i+1}: {exc}')
                # Release dependencies promptly; the failed test is still recorded as failed.
                first_events[i].set()
    report = {'status': 'failed' if errors else 'passed', 'test': args.mode, 'parameters': params,
              'model': model['id'], 'users': users, 'prepared_input_tokens': counts,
              'chat_template_token_offset': offset,
              'results': results, 'errors': errors, 'quality_evaluation': False}
    if not errors:
        metrics, failures = summarize(results, counts, minimums, 2 if args.mode == 'mixed' else users,
                                      args.mode == 'mixed')
        report['metrics'] = metrics; report['errors'].extend(failures)
        report['status'] = 'failed' if report['errors'] else 'passed'
    if clients[0].json('/health').get('status') != 'healthy':
        report['status'] = 'failed'; report['errors'].append('Backend unhealthy after benchmark')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'long', 'mixed'])
    parser.add_argument('--url')
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='max')
    parser.add_argument('--tokens', type=int, default=260000)
    parser.add_argument('--users', type=int, choices=range(1, 9))
    parser.add_argument('--max-output', type=int, default=4096)
    parser.add_argument('--min-output', type=int)
    parser.add_argument('--short-tokens', type=int, default=8192)
    parser.add_argument('--short-output', type=int, default=256)
    parser.add_argument('--timeout', type=int, default=7200, help='Wall-clock streaming budget in seconds')
    parser.add_argument('--corpus', type=Path, help='Optional representative UTF-8 text/code to repeat in the prompt')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/api-check.json')
    args = parser.parse_args()
    if min(args.tokens, args.max_output, args.short_tokens, args.short_output, args.timeout) < 1:
        parser.error('All limits must be positive')
    if args.min_output is not None and not 1 <= args.min_output <= args.max_output:
        parser.error('min-output must be between 1 and max-output')
    try:
        report = run_check(args)
    except Exception as exc:
        report = {'status': 'failed', 'test': args.mode, 'errors': [str(exc)]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'results', 'result'}}, indent=2))
    print(f'Saved {args.report}')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
