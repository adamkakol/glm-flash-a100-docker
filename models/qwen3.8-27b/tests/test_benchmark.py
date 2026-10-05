"""Real local HTTP/SSE tests. No GPU, weights, tokenizer package, or external API."""
import importlib.util
import json
from pathlib import Path
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODULE = Path(__file__).resolve().parents[1] / "qwen_bench/benchmark.py"
SPEC = importlib.util.spec_from_file_location("qwen_benchmark", MODULE)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def token_count(messages):
    # Deliberately unlike word counting. Client must ask this server and verify usage.
    return 7 + sum((len(str(message.get("content", ""))) + 3) // 4 for message in messages)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.paths.append(self.path)
            if self.path.endswith("tokenize"):
                self.server.tokenized.append(body)
                self.server.tokenize_sessions.append(self.headers.get("X-Session-ID"))
        if self.path == "/tokenize":
            payload = json.dumps({"count": token_count(body["messages"])}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        session_id = self.headers.get("X-Session-ID")
        with self.server.lock:
            backend = self.server.session_backends.setdefault(session_id, "worker" + str(1 + len(self.server.session_backends) % 2))
            if self.server.mode == "changed_backend":
                backend = "worker2" if backend == "worker1" else "worker1"
            self.server.session_requests.append({"session_id": session_id, "backend": backend, "body": body})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if self.server.mode != "missing_backend":
            self.send_header("X-Qwen-Backend", backend)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            if self.server.mode in ("comments_before_output", "comments_after_output"):
                if self.server.mode == "comments_after_output":
                    self.event({"choices": [{"delta": {"content": "first"}}]})
                for _ in range(100):
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    time.sleep(.01)
                return
            if self.server.mode == "slow_drip":
                for _ in range(100):
                    self.wfile.write(b"d")
                    self.wfile.flush()
                    time.sleep(.02)
                return
            if self.server.mode == "stall":
                time.sleep(.3)
                return
            if self.path == "/v1/responses":
                self.respond_native(body)
            else:
                self.chat(body)
        except (BrokenPipeError, ConnectionResetError):
            with self.server.lock:
                self.server.disconnects += 1

    def event(self, value):
        # Multiline JSON data, CRLF, comments, and fragmented TCP writes.
        content = json.dumps(value, indent=1, ensure_ascii=False)
        payload = b": heartbeat\r\n\r\n" + b"".join(b"data: " + line.encode() + b"\r\n" for line in content.splitlines()) + b"\r\n"
        for part in (payload[:11], payload[11:37], payload[37:]):
            self.wfile.write(part)
            self.wfile.flush()

    def chat(self, body):
        messages = body["messages"]
        content = messages[0]["content"]
        finish, calls = "stop", None
        if isinstance(content, list):
            value = {"colors": ["red", "blue"]} if any(p["type"] == "video_url" for p in content) else {"color": "red"}
            text = json.dumps(value)
        elif "BEGIN_MARKER=" in content:
            markers = re.findall(r"(?:BEGIN|MIDDLE|END)_MARKER=([0-9a-f]{16})", content)
            text = json.dumps(dict(zip(("begin", "middle", "end"), markers)))
            if self.server.mode == "wrong_answer":
                text = '{"begin":"incorrect"}'
        elif body.get("tools") and not any(m["role"] == "tool" for m in messages):
            calls = [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a":19,"b":23}'}}]
            text, finish = "", "tool_calls"
        elif body.get("tools"):
            self.server.chat_followup = messages[-1].get("tool_call_id") == "call_1"
            text = '{"answer":42}'
        else:
            text = '{"answer":14}'
        if body.get("min_tokens"):
            if self.server.mode == "queue_short" and body["max_tokens"] <= 64:
                time.sleep(.3)
            loops = 12 if body["max_tokens"] > 64 else 4
            for _ in range(loops):
                self.event({"choices": [{"delta": {"reasoning_content": "reasoning λ "}}]})
                time.sleep(.012)
            text, finish = "completed", "length"
        if self.server.mode == "length_smoke":
            finish = "length"
        if calls:
            self.event({"choices": [{"delta": {"tool_calls": calls}}]})
        else:
            midpoint = max(1, len(text) // 2)
            self.event({"choices": [{"delta": {"content": text[:midpoint]}}]})
            time.sleep(.025)
            self.event({"choices": [{"delta": {"content": text[midpoint:]}}]})
        count = token_count(messages) + (1 if self.server.mode == "wrong_usage" else 0)
        usage = {"prompt_tokens": count, "completion_tokens": body["max_tokens"] if body.get("min_tokens") else 20}
        if len(messages) > 1:
            usage["prompt_tokens_details"] = {"cached_tokens": count + 1 if self.server.mode == "invalid_cache_usage" else token_count(messages[:1])}
        tail = {"choices": [{"delta": {}, "finish_reason": None if self.server.mode == "no_finish" else finish}]}
        if self.server.mode != "no_usage":
            tail["usage"] = usage
        self.event(tail)
        if self.server.mode != "no_done":
            self.wfile.write(b"data: [DONE]\r\n\r\n")
            self.wfile.flush()

    def respond_native(self, body):
        followup = any(item.get("type") == "function_call_output" for item in body["input"])
        if followup:
            self.server.responses_followup = any(item.get("call_id") == "resp_call" and item.get("output") == "42" for item in body["input"])
            self.event({"type": "response.output_text.delta", "delta": '{"answer":42}'})
            final_text = '{"answer":43}' if self.server.mode == "response_text_mismatch" else '{"answer":42}'
            output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": final_text}]}]
        else:
            item = {"type": "function_call", "id": "fc_1", "call_id": "resp_call", "name": "add", "arguments": ""}
            if self.server.mode != "response_missing_added":
                self.event({"type": "response.output_item.added", "output_index": 0, "item": item})
            delta = '{"a":18,"b":23}' if self.server.mode == "response_args_mismatch" else '{"a":19,"b":23}'
            if self.server.mode != "response_missing_delta":
                self.event({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": delta[:6]})
                self.event({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": delta[6:]})
            item["arguments"] = '{"a":19,"b":23}'
            if self.server.mode == "response_changed_call_id":
                item["call_id"] = "different"
            if self.server.mode != "response_missing_done":
                self.event({"type": "response.output_item.done", "output_index": 0, "item": item})
            output = [item]
        kind = "response.incomplete" if self.server.mode == "incomplete_response" else "response.completed"
        self.event({"type": kind, "response": {"status": "completed", "output": output, "usage": {"input_tokens": 30, "output_tokens": 20}}})


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.lock = threading.Lock()
        self.server.paths, self.server.tokenized = [], []
        self.server.tokenize_sessions, self.server.session_requests, self.server.session_backends = [], [], {}
        self.server.mode, self.server.disconnects = "normal", 0
        self.server.responses_followup = self.server.chat_followup = False
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.client = benchmark.BenchmarkClient("http://127.0.0.1:%s/v1" % self.server.server_port,
                    "secret-test-key", timeout=3, idle_timeout=1, video_fixture_url="data:video/mp4;base64,fixture")

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        self.assertFalse(self.thread.is_alive())

    def test_exact_tokenizer_template_and_distinct_prompts(self):
        first = self.client.exact_prompt(256, 1)
        second = self.client.exact_prompt(256, 2)
        self.assertEqual(token_count(first), 256)
        self.assertNotEqual(first[0]["content"][:50], second[0]["content"][:50])
        self.assertEqual(self.server.paths[0], "/tokenize")
        self.assertTrue(self.server.tokenized[0]["add_generation_prompt"])
        self.assertEqual(self.server.tokenized[0]["chat_template_kwargs"]["reasoning_effort"], "xhigh")

    def test_mixed_concurrency_usage_and_safe_serialization(self):
        result = self.client.run_workload(256, 128, seed=22)
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["long_decode_overlap_s"], 0)
        self.assertTrue(result["short_decode_overlapped_all_long"])
        self.assertEqual([r["output_tokens"] for r in result["requests"]], [128, 128, 64])
        self.assertGreater(result["requests"][0]["output_tokens_per_second"], 100)
        serialized = json.dumps(result)
        self.assertNotIn("secret-test-key", serialized)
        self.assertNotIn("synthetic records", serialized)
        self.assertNotIn("reasoning λ", serialized)

    def test_followup_preserves_sessions_backends_and_grows_exact_chat(self):
        cold = self.client.run_workload(256, 128, seed=73)
        self.assertTrue(cold["passed"], cold)
        cold_requests = list(self.server.session_requests)
        warm = self.client.run_followup_workload(256, 96, seed=73, max_model_len=1024)
        self.assertTrue(warm["passed"], warm)
        self.assertEqual(warm["conversation_mode"], "synthetic_growing_followup")
        self.assertEqual(len(warm["requests"]), 2)
        self.assertFalse(warm["short_injected_after_long_decode"])
        self.assertGreater(warm["long_decode_overlap_s"], 0)
        for previous, current in zip(cold["requests"][:2], warm["requests"]):
            self.assertEqual(previous["session_id"], current["session_id"])
            self.assertEqual(previous["backend"], current["backend"])
            self.assertEqual(current["expected_backend"], previous["backend"])
            self.assertGreater(current["input_tokens"], 256)
            self.assertEqual(current["output_tokens"], 96)
            self.assertEqual(current["cached_input_tokens"], 256)
            self.assertNotIn("cached_input_tokens", previous)
            sent_cold = next(r for r in cold_requests if r["session_id"] == current["session_id"])
            sent_warm = next(r for r in self.server.session_requests[len(cold_requests):] if r["session_id"] == current["session_id"])
            self.assertEqual(sent_cold["body"]["messages"][0], sent_warm["body"]["messages"][0])
            self.assertEqual([m["role"] for m in sent_warm["body"]["messages"]], ["user", "assistant", "user"])
            self.assertEqual(current["input_tokens"], token_count(sent_warm["body"]["messages"]))
        self.assertTrue(all(value is None for value in self.server.tokenize_sessions))
        next_trial = self.client.run_workload(256, 128, seed=74, mixed=False)
        self.assertTrue(next_trial["passed"], next_trial)
        self.assertFalse({r["session_id"] for r in cold["requests"]} & {r["session_id"] for r in next_trial["requests"]})

    def test_followup_requires_successful_base_and_context_room(self):
        with self.assertRaisesRegex(benchmark.BenchmarkError, "successful matching cold"):
            self.client.run_followup_workload(256, 96, seed=75)
        self.assertTrue(self.client.run_workload(256, 128, seed=75, mixed=False)["passed"])
        count = len(self.server.session_requests)
        with self.assertRaisesRegex(benchmark.BenchmarkError, "context/output budget"):
            self.client.run_followup_workload(256, 96, seed=75, max_model_len=300)
        self.assertEqual(count, len(self.server.session_requests))

    def test_followup_rejects_invalid_cached_token_usage(self):
        self.assertTrue(self.client.run_workload(256, 128, mixed=False)["passed"])
        self.server.mode = "invalid_cache_usage"
        result = self.client.run_followup_workload(256, 96, max_model_len=1024)
        self.assertFalse(result["passed"])
        self.assertTrue(all("cached-token usage" in error for error in result["errors"].values()))

    def test_session_backend_identity_missing_or_changed_fails(self):
        self.server.mode = "missing_backend"
        result = self.client.run_workload(256, 128, mixed=False)
        self.assertFalse(result["passed"])
        self.assertTrue(all("X-Qwen-Backend" in error for error in result["errors"].values()))
        self.server.mode = "normal"
        cold = self.client.run_workload(256, 128, mixed=False)
        self.assertTrue(cold["passed"], cold)
        self.server.mode = "changed_backend"
        warm = self.client.run_followup_workload(256, 96, max_model_len=1024)
        self.assertFalse(warm["passed"])
        self.assertTrue(all("different backend" in error for error in warm["errors"].values()))
        repeated = self.client.run_workload(256, 128, mixed=False)
        self.assertFalse(repeated["passed"])

    def test_queued_short_request_fails(self):
        self.server.mode = "queue_short"
        result = self.client.run_workload(256, 128)
        self.assertFalse(result["passed"])
        self.assertFalse(result["short_decode_overlapped_all_long"])

    def test_usage_mismatch_and_terminal_validation(self):
        prompt = self.client.exact_prompt(128, 1)
        for mode in ("wrong_usage", "no_usage", "no_finish", "no_done"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                with self.assertRaises(benchmark.BenchmarkError):
                    self.client._stream("chat/completions", self.client._chat_payload(prompt, 64, stress=True), 128, True)

    def test_full_smoke_native_responses_and_disconnect(self):
        result = self.client.smoke()
        self.assertTrue(result["passed"], result)
        self.assertTrue(self.server.responses_followup)
        self.assertTrue(self.server.chat_followup)
        self.assertTrue(result["gates"]["cancellation"]["post_cancel_request_passed"])
        self.assertIn("/v1/responses", self.server.paths)

    def test_missing_video_fails_qualification(self):
        self.client.video_fixture_url = None
        result = self.client.smoke()
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["video"]["passed"])

    def test_smoke_length_and_incomplete_responses_fail(self):
        self.server.mode = "length_smoke"
        result = self.client.smoke()
        self.assertFalse(result["gates"]["text"]["passed"])
        self.server.mode = "incomplete_response"
        result = self.client.smoke()
        self.assertFalse(result["gates"]["responses"]["passed"])

    def test_long_retrieval_exact_count_and_known_answer(self):
        result = self.client.long_retrieval(256, seed=43)
        self.assertTrue(result["passed"], result)
        self.assertTrue(all(r["correct"] and r["input_tokens"] == 256 for r in result["requests"]))
        self.assertNotIn("MARKER", json.dumps(result))
        self.server.mode = "wrong_answer"
        self.assertFalse(self.client.long_retrieval(256)["passed"])

    def test_idle_timeout_is_bounded(self):
        self.server.mode = "stall"
        self.client.idle_timeout = .05
        started = time.monotonic()
        with self.assertRaises((TimeoutError, OSError)):
            self.client._stream("chat/completions", self.client._chat_payload([{"role": "user", "content": "test"}], 10))
        self.assertLess(time.monotonic() - started, .5)

    def test_total_timeout_bounds_partial_line_drip(self):
        self.server.mode = "slow_drip"
        self.client.timeout = .12
        self.client.idle_timeout = 1
        started = time.monotonic()
        with self.assertRaises((benchmark.BenchmarkError, OSError, ValueError)):
            self.client._stream("chat/completions", self.client._chat_payload([{"role": "user", "content": "test"}], 10))
        self.assertLess(time.monotonic() - started, .5)

    def test_latency_budget_fails_completion(self):
        self.client.budgets["max_useful_gap_s"] = .001
        result = self.client.run_workload(128, 128, mixed=False)
        self.assertFalse(result["passed"])
        self.assertTrue(result["errors"] or result["requests"])

    def test_comments_do_not_reset_useful_deadline(self):
        self.client.budgets.update(ttft_s=.06, max_useful_gap_s=.06)
        for mode in ("comments_before_output", "comments_after_output"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                started = time.monotonic()
                with self.assertRaises((benchmark.BenchmarkError, OSError, ValueError)):
                    self.client._stream("chat/completions", self.client._chat_payload([{"role": "user", "content": "test"}], 10))
                self.assertLess(time.monotonic() - started, .4)

    def test_responses_stream_and_final_must_agree(self):
        for mode in ("response_args_mismatch", "response_text_mismatch", "response_missing_added",
                     "response_missing_delta", "response_missing_done", "response_changed_call_id"):
            with self.subTest(mode=mode):
                self.server.mode = mode
                result = self.client.smoke()
                self.assertFalse(result["gates"]["responses"]["passed"], result)


if __name__ == "__main__":
    unittest.main()
