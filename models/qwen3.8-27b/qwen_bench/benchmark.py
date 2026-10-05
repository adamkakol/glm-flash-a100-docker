"""Dependency-free, server-tokenized qualification. Never persists request content."""
from __future__ import annotations

import base64
import http.client
import json
import random
import socket
import struct
import threading
import time
import urllib.parse
import zlib


class BenchmarkError(RuntimeError):
    pass


def sse_events(response, before_read=lambda: None):
    """Parse SSE across arbitrary transport fragments, including multiline data."""
    data, event = [], "message"
    while True:
        before_read()
        raw = response.readline(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise BenchmarkError("SSE line exceeds safety bound")
        if not raw:
            if data:
                yield event, "\n".join(data)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            data, event = [], "message"
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                data.append(value)
            elif field == "event":
                event = value


def image_fixture():
    """A small lossless red PNG, generated in memory using only the stdlib."""
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    size = 64
    pixels = (b"\0" + b"\xff\0\0" * size) * size
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


class BenchmarkClient:
    def __init__(self, base_url, api_key="", model="qwen3.8-27b", timeout=7200,
                 idle_timeout=300, progress=None, latency_budgets=None,
                 video_fixture_url=None):
        url = urllib.parse.urlsplit(base_url.rstrip("/"))
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("base_url must be an HTTP(S) URL without credentials")
        if url.query or url.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        self.url, self.api_key, self.model = url, api_key, model
        self.timeout, self.idle_timeout = float(timeout), float(idle_timeout)
        if min(self.timeout, self.idle_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        self.progress = progress or (lambda event: None)
        self.budgets = {"ttft_s": 300, "max_useful_gap_s": 120}
        self.budgets.update(latency_budgets or {})
        self.video_fixture_url = video_fixture_url
        self._lock, self._connections = threading.Lock(), set()

    def _notify(self, **event):
        # Callbacks receive metrics and stage names, never prompts or credentials.
        self.progress(event)

    @staticmethod
    def _error(exc):
        return type(exc).__name__ + (": " + str(exc) if isinstance(exc, BenchmarkError) else "")

    @staticmethod
    def _abort(connection):
        # shutdown wakes a blocked buffered readline even when HTTPConnection
        # detached its socket for a Connection: close response.
        sock = getattr(connection, "_benchmark_socket", None) or connection.sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        connection.close()

    def close(self):
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            self._abort(connection)

    def _path(self, endpoint):
        base = self.url.path.rstrip("/")
        if endpoint == "tokenize":
            return (base[:-3] if base.endswith("/v1") else base) + "/tokenize"
        return (base if base.endswith("/v1") else base + "/v1") + "/" + endpoint

    def _arm_useful_deadline(self, connection, deadline):
        # One sleeping watchdog per connection; do not spawn a timer per token.
        if not hasattr(connection, "_benchmark_useful_condition"):
            connection._benchmark_useful_condition = threading.Condition()
            connection._benchmark_useful_stopped = False
            connection._benchmark_useful_expired = False
            connection._benchmark_useful_deadline = deadline
            def watch():
                condition = connection._benchmark_useful_condition
                with condition:
                    while not connection._benchmark_useful_stopped:
                        remaining = connection._benchmark_useful_deadline - time.monotonic()
                        if remaining <= 0:
                            connection._benchmark_useful_expired = True
                            break
                        condition.wait(remaining)
                    expired = connection._benchmark_useful_expired
                if expired:
                    self._abort(connection)
            watcher = threading.Thread(target=watch, daemon=True)
            connection._benchmark_useful_watcher = watcher
            watcher.start()
        with connection._benchmark_useful_condition:
            if connection._benchmark_useful_expired:
                raise BenchmarkError("useful output deadline exceeded")
            connection._benchmark_useful_deadline = deadline
            connection._benchmark_useful_condition.notify_all()

    def _open(self, endpoint, payload, deadline, useful_deadline=None):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        connect_deadline = min(deadline, useful_deadline) if useful_deadline is not None else deadline
        connection = cls(self.url.hostname, self.url.port,
                         timeout=min(self.idle_timeout, max(.001, connect_deadline - time.monotonic())))
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        with self._lock:
            self._connections.add(connection)
        timer = threading.Timer(max(.001, deadline - time.monotonic()), self._abort, args=(connection,))
        timer.daemon = True
        connection._benchmark_timer = timer
        timer.start()
        try:
            if useful_deadline is not None:
                self._arm_useful_deadline(connection, useful_deadline)
            connection.connect()
            connection._benchmark_socket = connection.sock
            if getattr(connection, "_benchmark_useful_expired", False):
                raise BenchmarkError("first useful output deadline exceeded")
            connection.request("POST", self._path(endpoint), json.dumps(payload).encode(), headers)
            response = connection.getresponse()
            connection._benchmark_response = response
            if response.status != 200:
                raise BenchmarkError("HTTP status %d from %s" % (response.status, endpoint))
            return connection, response
        except BaseException:
            self._release(connection)
            raise

    def _release(self, connection):
        timer = getattr(connection, "_benchmark_timer", None)
        if timer:
            timer.cancel()
        condition = getattr(connection, "_benchmark_useful_condition", None)
        if condition:
            with condition:
                connection._benchmark_useful_stopped = True
                condition.notify_all()
        response = getattr(connection, "_benchmark_response", None)
        if response:
            response.close()
        connection.close()
        with self._lock:
            self._connections.discard(connection)

    def _deadline(self, connection, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BenchmarkError("request exceeded total timeout")
        sock = getattr(connection, "_benchmark_socket", None) or connection.sock
        if sock:
            sock.settimeout(min(self.idle_timeout, remaining))

    def tokenize(self, messages, reasoning_effort="xhigh"):
        payload = {"model": self.model, "messages": messages, "add_generation_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning_effort}}
        deadline = time.monotonic() + self.timeout
        connection, response = self._open("tokenize", payload, deadline)
        try:
            self._deadline(connection, deadline)
            data = json.loads(response.read())
            count = data.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise BenchmarkError("/tokenize did not return a positive count")
            return count
        finally:
            self._release(connection)

    def exact_prompt(self, target, seed, reasoning_effort="xhigh", retrieval=False):
        """Find an exact template-inclusive size; fail rather than truncate or estimate."""
        rng = random.Random(seed)
        # The unique initial tag makes prefix reuse across users unlikely.
        tag = "%032x" % rng.getrandbits(128)
        prefix = "Case " + tag + ": Review the following synthetic records.\n"
        suffix = "\nExplain the records and produce a detailed Python validation implementation."
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        # One bounded buffer; counting is delegated to the deployed tokenizer/template.
        buffer = " ".join("".join(rng.choices(alphabet, k=9)) for _ in range(max(32, target)))
        markers = ["%016x" % rng.getrandbits(64) for _ in range(3)]
        def messages(n, extra=""):
            if retrieval:
                body = ("BEGIN_MARKER=" + markers[0] + "\n" + buffer[:n // 2]
                        + "\nMIDDLE_MARKER=" + markers[1] + "\n" + buffer[n // 2:n]
                        + extra + "\nEND_MARKER=" + markers[2]
                        + '\nReturn only JSON with keys begin, middle, end, containing the respective marker values verbatim.')
                content = prefix + body
            else:
                content = prefix + buffer[:n] + extra + suffix
            return [{"role": "user", "content": content}]
        low, high, best = 0, len(buffer), None
        for _ in range(28):
            if low > high:
                break
            middle = (low + high) // 2
            value = messages(middle)
            count = self.tokenize(value, reasoning_effort)
            if count == target:
                return value
            if count < target:
                best = (middle, count)
                low = middle + 1
            else:
                high = middle - 1
        if best:
            length, count = best
            # BPE boundaries need not be perfectly monotonic; try several unit fillers.
            for filler in (" x", "\n", " 0", " z", ". "):
                for units in range(max(0, target - count - 3), target - count + 5):
                    value = messages(length, filler * units)
                    if self.tokenize(value, reasoning_effort) == target:
                        return value
        raise BenchmarkError("cannot construct requested exact token count; no request was truncated")

    def _chat_payload(self, messages, output_tokens, reasoning_effort="xhigh", stress=False, **extra):
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}, "max_tokens": output_tokens,
                   "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": reasoning_effort}}
        if stress:
            payload["min_tokens"] = output_tokens
        payload.update(extra)
        return payload

    def _stream(self, endpoint, payload, expected_input=None, stress=False,
                on_first=None, cancel_after_first=False):
        started = time.monotonic()
        deadline = started + (min(self.timeout, 30) if cancel_after_first else self.timeout)
        first_deadline = min(deadline, started + self.budgets.get("ttft_s", self.timeout))
        connection, response = self._open(endpoint, payload, deadline, first_deadline)
        response_calls = {}
        first, last, gaps, usage, finish = None, None, [], None, None
        text, tools, outputs, done, terminal, cancelled = [], {}, [], False, False, False
        try:
            content_type = response.getheader("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise BenchmarkError("stream response is not text/event-stream")
            for event, raw in sse_events(response, lambda: self._deadline(connection, deadline)):
                if raw == "[DONE]":
                    done = True
                    break
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    raise BenchmarkError("invalid SSE JSON event") from exc
                if data.get("error") or data.get("type") in ("error", "response.failed", "response.incomplete"):
                    raise BenchmarkError("server reported an unsuccessful stream")
                useful = False
                if endpoint == "chat/completions":
                    if data.get("usage"):
                        usage = data["usage"]
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text.append(delta["content"])
                            useful = True
                        useful = useful or bool(delta.get("reasoning_content") or delta.get("reasoning"))
                        for tool in delta.get("tool_calls") or []:
                            index = tool.get("index", 0)
                            current = tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            current["id"] += tool.get("id") or ""
                            function = tool.get("function") or {}
                            for key in ("name", "arguments"):
                                current[key] += function.get(key) or ""
                            useful = useful or bool(tool.get("id") or function.get("name") or function.get("arguments"))
                        if choice.get("finish_reason"):
                            finish = choice["finish_reason"]
                else:
                    kind = data.get("type", event)
                    if kind in ("response.output_text.delta", "response.reasoning_text.delta",
                                "response.reasoning_summary_text.delta", "response.function_call_arguments.delta") and data.get("delta"):
                        useful = True
                        if kind == "response.output_text.delta":
                            text.append(data["delta"])
                    if kind == "response.output_item.added" and (data.get("item") or {}).get("type") == "function_call":
                        item = data["item"]
                        item_id = item.get("id")
                        if not item_id or item_id in response_calls or not item.get("call_id") or not item.get("name"):
                            raise BenchmarkError("Responses function item lacks a unique id, call_id, or name")
                        if item.get("arguments"):
                            raise BenchmarkError("Responses function item must begin before streaming arguments")
                        response_calls[item_id] = {"call_id": item["call_id"], "name": item["name"],
                            "index": data.get("output_index"), "arguments": "", "delta_seen": False, "done": False}
                    if kind == "response.function_call_arguments.delta":
                        call = response_calls.get(data.get("item_id"))
                        if not call or call["done"] or data.get("output_index") != call["index"] or not isinstance(data.get("delta"), str):
                            raise BenchmarkError("Responses argument delta has invalid item identity or ordering")
                        call["arguments"] += data["delta"]
                        call["delta_seen"] = call["delta_seen"] or bool(data["delta"])
                    if kind == "response.output_item.done":
                        item = data.get("item") or {}
                        if item.get("type") == "function_call":
                            call = response_calls.get(item.get("id"))
                            if (not call or call["done"] or not call["delta_seen"] or data.get("output_index") != call["index"]
                                    or any(item.get(key) != call[key] for key in ("call_id", "name", "arguments"))):
                                raise BenchmarkError("Responses completed function differs from its streamed arguments or identity")
                            call["done"] = True
                        outputs.append(item)
                    if kind == "response.completed":
                        result = data.get("response") or {}
                        if result.get("status") != "completed":
                            raise BenchmarkError("Responses terminal status is not completed")
                        usage = result.get("usage")
                        outputs = result.get("output")
                        if not isinstance(outputs, list):
                            raise BenchmarkError("Responses completed event lacks its output array")
                        final_calls = {item.get("id"): item for item in outputs if item.get("type") == "function_call"}
                        if set(final_calls) != set(response_calls) or len(final_calls) != sum(item.get("type") == "function_call" for item in outputs):
                            raise BenchmarkError("Responses final function items differ from streamed items")
                        for item_id, call in response_calls.items():
                            if not call["done"] or any(final_calls[item_id].get(key) != call[key] for key in ("call_id", "name", "arguments")):
                                raise BenchmarkError("Responses final function differs from its streamed lifecycle")
                        final_text = "".join(part.get("text", "") for item in outputs if item.get("type") == "message"
                                             for part in item.get("content", []) if part.get("type") == "output_text")
                        if final_text != "".join(text):
                            raise BenchmarkError("Responses final text differs from streamed text")
                        finish, terminal = "stop", True
                if useful:
                    now = time.monotonic()
                    self._arm_useful_deadline(connection, min(deadline, now + self.budgets.get("max_useful_gap_s", self.timeout)))
                    if first is None:
                        first = now
                        if on_first:
                            on_first(now)
                    if last is not None:
                        gaps.append(now - last)
                    last = now
                    if cancel_after_first:
                        cancelled = True
                        break
                if terminal:
                    done = True
                    break
            ended = time.monotonic()
            if getattr(connection, "_benchmark_useful_expired", False):
                raise BenchmarkError("useful output deadline exceeded")
            if cancelled:
                return {"cancelled": True, "duration_s": ended - started}
            if not done or not finish or first is None:
                raise BenchmarkError("stream lacked useful output, finish reason, or terminal event")
            if finish not in (("length", "stop") if stress else ("stop", "tool_calls")):
                raise BenchmarkError("unexpected finish reason")
            if not isinstance(usage, dict):
                raise BenchmarkError("stream did not include terminal token usage")
            input_key, output_key = (("prompt_tokens", "completion_tokens") if endpoint == "chat/completions"
                                     else ("input_tokens", "output_tokens"))
            count_in, count_out = usage.get(input_key), usage.get(output_key)
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (count_in, count_out)):
                raise BenchmarkError("stream usage token counts are missing or invalid")
            if expected_input is not None and count_in != expected_input:
                raise BenchmarkError("server input usage differs from /tokenize; possible truncation or template mismatch")
            if stress and count_out != payload["max_tokens"]:
                raise BenchmarkError("stress output token count differs from requested budget")
            gaps.append(ended - last)
            elapsed_decode = last - first
            result = {"input_tokens": count_in, "output_tokens": count_out, "finish_reason": finish,
                      "ttft_s": first - started, "max_useful_gap_s": max(gaps, default=0.0),
                      "p95_useful_gap_s": sorted(gaps)[max(0, int(len(gaps) * .95 + .9999) - 1)] if gaps else 0.0,
                      "duration_s": ended - started,
                      "output_tokens_per_second": (count_out - 1) / elapsed_decode if elapsed_decode > 0 else None,
                      "end_to_end_tokens_per_second": count_out / (ended - started),
                      "started_at": started, "first_useful_at": first, "last_useful_at": last, "ended_at": ended,
                      "text": "".join(text), "tool_calls": list(tools.values()), "output": outputs}
            violations = []
            for metric in ("ttft_s", "max_useful_gap_s", "duration_s"):
                if metric in self.budgets and result[metric] > self.budgets[metric]:
                    violations.append(metric)
            result["latency_violations"], result["passed"] = violations, not violations
            return result
        finally:
            self._release(connection)

    @staticmethod
    def _public(result):
        return {k: v for k, v in result.items() if k not in ("text", "tool_calls", "output")}

    def run_workload(self, input_tokens, output_tokens, concurrency=2, mixed=True,
                     seed=1, reasoning_effort="xhigh"):
        if min(input_tokens, output_tokens, concurrency) < 1:
            raise ValueError("token budgets and concurrency must be positive")
        prompts = [self.exact_prompt(input_tokens, seed + i * 104729, reasoning_effort) for i in range(concurrency)]
        short_target = min(512, input_tokens)
        short_prompt = self.exact_prompt(short_target, seed + 999983, reasoning_effort) if mixed else None
        self._notify(stage="prompts_tokenized", input_tokens=input_tokens, concurrency=concurrency)
        barrier = threading.Barrier(concurrency + 1)
        decoding = [threading.Event() for _ in prompts]
        results, failures, lock = {}, {}, threading.Lock()
        def worker(index, messages, target, output, synchronized=False):
            try:
                if synchronized:
                    barrier.wait(timeout=min(30, self.timeout))
                def first(_):
                    if synchronized:
                        decoding[index].set()
                    self._notify(stage="decoding", request=index)
                result = self._stream("chat/completions", self._chat_payload(messages, output, reasoning_effort, True),
                                      target, True, first)
                with lock:
                    results[index] = result
                self._notify(stage="request_completed", request=index, metrics=self._public(result))
            except Exception as exc:
                with lock:
                    failures[index] = self._error(exc)
                self._notify(stage="request_failed", request=index, error=type(exc).__name__)
        threads = [threading.Thread(target=worker, args=(i, prompt, input_tokens, output_tokens, True), daemon=True)
                   for i, prompt in enumerate(prompts)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=min(30, self.timeout))
        deadline = time.monotonic() + self.timeout + 2
        short_injected = False
        while any(thread.is_alive() for thread in threads):
            if mixed and not short_injected and all(event.is_set() for event in decoding):
                short_injected = True
                thread = threading.Thread(target=worker, args=(concurrency, short_prompt, short_target, min(64, output_tokens)), daemon=True)
                threads.append(thread)
                thread.start()
            if time.monotonic() >= deadline:
                self.close()
                break
            for thread in threads:
                thread.join(timeout=.01)
        for thread in threads:
            thread.join(timeout=min(1, self.idle_timeout))
        if any(thread.is_alive() for thread in threads):
            failures[-1] = "TimeoutError: workers did not finish within total deadline"
        long = [results[i] for i in range(concurrency) if i in results]
        overlap = (max(0.0, min(r["last_useful_at"] for r in long) - max(r["first_useful_at"] for r in long))
                   if len(long) == concurrency else 0.0)
        short = results.get(concurrency)
        short_overlap = bool(short and long and short["first_useful_at"] < min(r["last_useful_at"] for r in long)
                             and short["first_useful_at"] >= max(r["first_useful_at"] for r in long))
        passed = (not failures and len(long) == concurrency and all(r["passed"] for r in results.values())
                  and (concurrency == 1 or overlap > 0) and (not mixed or short_overlap))
        return {"passed": bool(passed), "stress_min_tokens": True, "seed": seed,
                "requested_input_tokens": input_tokens, "requested_output_tokens": output_tokens,
                "concurrency": concurrency, "reasoning_effort": reasoning_effort,
                "requests": [dict(request=i, **self._public(result)) for i, result in sorted(results.items())],
                "errors": {str(k): v for k, v in failures.items()}, "long_decode_overlap_s": overlap,
                "short_injected_after_long_decode": short_injected, "short_decode_overlapped_all_long": short_overlap}

    def long_retrieval(self, input_tokens, seed=1, concurrency=2, reasoning_effort="xhigh"):
        """Separate correctness gate; never uses forced minimum generation length."""
        import re
        if min(input_tokens, concurrency) < 1:
            raise ValueError("input tokens and concurrency must be positive")
        prompts = [self.exact_prompt(input_tokens, seed + i * 104729, reasoning_effort, retrieval=True)
                   for i in range(concurrency)]
        barrier = threading.Barrier(concurrency + 1)
        results, lock = {}, threading.Lock()
        def worker(index, messages):
            try:
                expected = dict(zip(("begin", "middle", "end"),
                    re.findall(r"(?:BEGIN|MIDDLE|END)_MARKER=([0-9a-f]{16})", messages[0]["content"])))
                barrier.wait(timeout=min(30, self.timeout))
                result = self._stream("chat/completions", self._chat_payload(messages, 4096, reasoning_effort), input_tokens)
                correct = result["finish_reason"] == "stop" and json.loads(result["text"].strip()) == expected
                value = {**self._public(result), "correct": correct, "passed": result["passed"] and correct}
            except Exception as exc:
                value = {"passed": False, "error": self._error(exc)}
            with lock:
                results[index] = value
        threads = [threading.Thread(target=worker, args=(i, prompt), daemon=True) for i, prompt in enumerate(prompts)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=min(30, self.timeout))
        deadline = time.monotonic() + self.timeout + 2
        while any(thread.is_alive() for thread in threads) and time.monotonic() < deadline:
            for thread in threads:
                thread.join(timeout=.01)
        if any(thread.is_alive() for thread in threads):
            self.close()
            for thread in threads:
                thread.join(timeout=min(1, self.idle_timeout))
        successful = [r for r in results.values() if "first_useful_at" in r]
        overlap = (max(0.0, min(r["last_useful_at"] for r in successful) - max(r["first_useful_at"] for r in successful))
                   if len(successful) == concurrency else 0.0)
        return {"passed": len(results) == concurrency and all(r["passed"] for r in results.values()),
                "requested_input_tokens": input_tokens, "concurrency": concurrency, "seed": seed,
                "long_decode_overlap_s": overlap,
                "requests": [dict(request=i, **value) for i, value in sorted(results.items())]}

    def smoke(self):
        """Real model behavior gates; unsupported/missing modalities fail qualification."""
        gates = {}
        def gate(name, operation):
            self._notify(stage="smoke", gate=name)
            try:
                result = operation()
                gates[name] = {"passed": True, **(result or {})}
            except Exception as exc:
                gates[name] = {"passed": False, "error": self._error(exc)}
        def chat(messages, **extra):
            result = self._stream("chat/completions", self._chat_payload(messages, 2048, "low", **extra))
            if not result["passed"]:
                raise BenchmarkError("smoke latency budget exceeded")
            return result
        def answer(result, value):
            try:
                actual = json.loads(result["text"].strip())
            except ValueError as exc:
                raise BenchmarkError("model did not return the required JSON answer") from exc
            if actual != value or result["finish_reason"] != "stop":
                raise BenchmarkError("incorrect known-answer completion")
            return self._public(result)
        gate("text", lambda: answer(chat([{"role": "user", "content":
            'Evaluate Python sum(x*x for x in range(4)). Return only JSON with key answer and its integer value.'}]), {"answer": 14}))
        gate("image", lambda: answer(chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_fixture()}},
            {"type": "text", "text": 'Identify the dominant color. Return only JSON with key color and its lowercase name.'}]}]), {"color": "red"}))
        def video():
            if not self.video_fixture_url:
                raise BenchmarkError("video qualification requires a short fixture: solid red then solid blue; supply video_fixture_url")
            result = chat([{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": self.video_fixture_url}},
                {"type": "text", "text": 'List the dominant colors in temporal order. Return only JSON with key colors and an array of lowercase color names.'}]}])
            return answer(result, {"colors": ["red", "blue"]})
        gate("video", video)
        schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                  "required": ["a", "b"], "additionalProperties": False}
        function = {"name": "add", "description": "Add two integers using the tool.", "parameters": schema}
        prompt = "You must call the add tool with a=19 and b=23. After receiving its result, return only JSON with key answer and the numeric result."
        def tool():
            messages = [{"role": "user", "content": prompt}]
            result = chat(messages, tools=[{"type": "function", "function": function}], tool_choice="auto")
            calls = result["tool_calls"]
            if result["finish_reason"] != "tool_calls" or len(calls) != 1:
                raise BenchmarkError("chat did not finish with exactly one tool call")
            call = calls[0]
            if not call["id"] or call["name"] != "add" or json.loads(call["arguments"]) != {"a": 19, "b": 23}:
                raise BenchmarkError("chat tool call has incorrect id, name, or arguments")
            messages += [{"role": "assistant", "content": None, "tool_calls": [{"id": call["id"], "type": "function",
                          "function": {"name": call["name"], "arguments": call["arguments"]}}]},
                         {"role": "tool", "tool_call_id": call["id"], "content": "42"}]
            return answer(chat(messages, tools=[{"type": "function", "function": function}], tool_choice="auto"), {"answer": 42})
        gate("tool", tool)
        def responses():
            inputs = [{"role": "user", "content": prompt}]
            payload = {"model": self.model, "input": inputs, "stream": True, "store": False, "max_output_tokens": 2048,
                       "tools": [{"type": "function", **function, "strict": True}], "tool_choice": "auto"}
            result = self._stream("responses", payload)
            calls = [item for item in result["output"] if item.get("type") == "function_call"]
            if len(calls) != 1 or not calls[0].get("call_id") or calls[0].get("name") != "add":
                raise BenchmarkError("Responses did not emit the expected function call")
            call = calls[0]
            if json.loads(call["arguments"]) != {"a": 19, "b": 23}:
                raise BenchmarkError("Responses tool arguments are incorrect")
            payload["input"] = inputs + result["output"] + [{"type": "function_call_output", "call_id": call["call_id"], "output": "42"}]
            followup = self._stream("responses", payload)
            if not result["passed"] or not followup["passed"]:
                raise BenchmarkError("Responses latency budget exceeded")
            return answer(followup, {"answer": 42})
        gate("responses", responses)
        def cancellation():
            payload = self._chat_payload([{"role": "user", "content": "Write a long Python tutorial."}], 4096, "low", True)
            result = self._stream("chat/completions", payload, cancel_after_first=True)
            if not result.get("cancelled"):
                raise BenchmarkError("request did not reach cancellable streaming output")
            # A fresh request proves service remains available after client disconnect.
            answer(chat([{"role": "user", "content": 'Evaluate Python sum(x*x for x in range(4)). Return only JSON with key answer and its integer value.'}]), {"answer": 14})
            return {**result, "post_cancel_request_passed": True,
                    "limitation": "client disconnect verified; inspect server metrics to confirm GPU work was reclaimed"}
        gate("cancellation", cancellation)
        return {"passed": all(value["passed"] for value in gates.values()), "gates": gates}
