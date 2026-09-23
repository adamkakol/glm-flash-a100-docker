#!/usr/bin/env python3
"""Smoke test or two-user long-context capacity check; host Python, no packages needed."""
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

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, url, key, timeout):
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout

    def open(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read(4096).decode(errors='replace')}") from exc

    def json(self, path, body=None):
        with self.open(path, body) as response:
            return json.load(response)

    def token_count(self, messages):
        result = self.json("/v1/token/encode", {"text": messages,
                                               "template_vars": {"reasoning_effort": "low"}})
        return result["length"]


def make_prompt(client, target, max_input):
    nonce = uuid.uuid4().hex
    rng = random.Random(nonce)
    # Different early prefixes/data prevent the two requests sharing one large KV prefix.
    records = "\n".join(f"Record {i}: {rng.getrandbits(128):032x}." for i in range(target // 8 + 100))
    prefix = f"Request {nonce}. Read these synthetic records. This is a capacity test.\n"
    suffix = "\nAfter reading the records, reply with ACK."

    def messages(length):
        return [{"role": "user", "content": prefix + records[:length] + suffix}]

    low, high = 0, min(len(records), target * 3)
    count = client.token_count(messages(high))
    while count < target:
        if high == len(records):
            raise RuntimeError("Unable to build a sufficiently long prompt.")
        low, high = high, min(len(records), high * 2)
        count = client.token_count(messages(high))
    # Token count need not be perfectly monotonic around subword boundaries.
    # Always retain a measured candidate that meets the lower bound.
    best = messages(high)
    best_count = count
    for _ in range(16):
        if target <= best_count <= min(target + 512, max_input):
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
    raise RuntimeError("Could not fit the measured prompt into the configured context.")


def stream_request(client, model, messages, max_tokens, gate, number):
    gate.wait()
    start = time.time()
    first = None
    usage = None
    finish = None
    output = ""
    done = False
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    with client.open("/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if text and first is None:
                    first = time.time()
                if len(output) < 240:
                    output += text[:240 - len(output)]
                finish = choice.get("finish_reason") or finish
    end = time.time()
    if not done or not usage or usage.get("completion_tokens", 0) < 1 or first is None:
        raise RuntimeError("Incomplete stream or missing token usage/output.")
    if finish not in {"stop", "length", "tool_calls"}:
        raise RuntimeError(f"Unexpected completion status: {finish}")
    return {"request": number, "started_at": start, "first_token_at": first, "finished_at": end,
            "time_to_first_token_s": round(first - start, 3), "total_s": round(end - start, 3),
            "finish_reason": finish, "usage": usage, "output_excerpt": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["smoke", "long"])
    parser.add_argument("--url", help="API origin, without /v1; defaults to the port in .env")
    parser.add_argument("--tokens", type=int, default=250000, help="Minimum input tokens per long-test request")
    parser.add_argument("--users", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=3600, help="Socket read timeout, seconds")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/api-check.json")
    args = parser.parse_args()
    if args.tokens < 1 or args.max_output < 1:
        parser.error("Token limits must be positive.")
    env = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().splitlines() if "=" in line and not line.startswith("#"))
    url = args.url or f"http://127.0.0.1:{env.get('API_PORT', '5000')}"
    keys = json.loads((ROOT / "secrets/api_tokens.yml").read_text())["api_key"]
    if isinstance(keys, str):
        keys = [keys]
    clients = [Client(url, keys[i % len(keys)], args.timeout) for i in range(args.users)]
    model = clients[0].json("/v1/model")
    params = model["parameters"]
    if params["max_seq_len"] < 262144 or params["cache_size"] < 2 * params["max_seq_len"] or params["max_batch_size"] != 2:
        raise RuntimeError("Server has not loaded the expected two-user long-context settings.")
    if args.mode == "smoke":
        result = clients[0].json("/v1/chat/completions", {
            "model": model["id"], "messages": [{"role": "user", "content": "Reply with READY."}],
            "max_tokens": args.max_output, "temperature": 1.0, "top_p": 0.95,
            "chat_template_kwargs": {"reasoning_effort": "low"}})
        message = result["choices"][0]["message"]
        if not (message.get("content") or message.get("reasoning_content")):
            raise RuntimeError("Server returned no generated text.")
        print(json.dumps(result, indent=2))
        report = {"test": "smoke", "model": model["id"], "parameters": params, "result": result}
    else:
        max_input = params["max_seq_len"] - args.max_output - 2048
        if args.tokens > max_input:
            raise RuntimeError("Input + requested output + checkpoint margin exceeds the context window.")
        prompts = []
        for i, client in enumerate(clients):
            messages, count = make_prompt(client, args.tokens, max_input)
            prompts.append(messages)
            print(f"Prepared request {i + 1}: {count:,} input tokens", flush=True)
        gate = threading.Barrier(args.users)
        with ThreadPoolExecutor(max_workers=args.users) as pool:
            futures = [pool.submit(stream_request, client, model["id"], prompts[i],
                                   args.max_output, gate, i + 1) for i, client in enumerate(clients)]
            results = [future.result() for future in futures]
        for result in results:
            if result["usage"]["prompt_tokens"] < args.tokens:
                raise RuntimeError("The server reported fewer input tokens than requested; capacity test failed.")
        overlap = max(0, min(r["finished_at"] for r in results) - max(r["first_token_at"] for r in results))
        report = {"test": "long-context capacity, not model quality", "model": model["id"],
                  "parameters": params, "users": args.users, "minimum_input_tokens": args.tokens,
                  "generation_intervals_overlap_s": round(overlap, 3) if args.users == 2 else None,
                  "results": results}
        print(json.dumps(report, indent=2))
        if args.users == 2 and overlap == 0:
            print("Both HTTP requests completed, but their generation intervals did not overlap. This run does not demonstrate concurrent decoding; inspect scheduling or retest with longer outputs.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
