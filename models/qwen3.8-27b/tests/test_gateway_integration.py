"""Opt-in real HAProxy checks against local mock workers; no model or GPU needed.

Pull the reviewed image explicitly, then set QWEN_TEST_HAPROXY_IMAGE to the exact
gateway.DEFAULT_IMAGE digest. Tests never pull images and remove their own
containers and temporary files. The caller owns image cleanup after testing.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_bench.gateway import DEFAULT_IMAGE, Gateway

IMAGE = os.environ.get("QWEN_TEST_HAPROXY_IMAGE", "")
SECRET = "gateway-integration-only-not-a-real-secret"


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class MockWorker(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), MockHandler)
        self.calls = []
        self.lock = threading.Lock()
        self.hold_started = threading.Event()
        self.hold_release = threading.Event()
        self.sse_first = threading.Event()
        self.sse_release = threading.Event()
        self.sse_second = threading.Event()
        self.worker_name = f"worker-{self.server_port}"
        self.healthy = True
        self.incoming_backend_headers = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_port}"


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            # HAProxy may close a health connection as soon as it sees status 200.
            pass

    def log_message(self, *_):
        pass

    def reply(self, status, content):
        body = json.dumps(content).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Qwen-Backend", "spoofed-worker")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200 if self.server.healthy else 503, {"healthy": self.server.healthy})
            return
        self.handle_model_request("models")

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.handle_model_request(json.loads(data or b"{}").get("mode", "echo"))

    def handle_model_request(self, mode):
        if self.headers.get("Authorization") != f"Bearer {SECRET}":
            self.reply(401, {"error": "backend authentication failed"})
            return
        with self.server.lock:
            self.server.calls.append(mode)
            self.server.incoming_backend_headers.append(self.headers.get("X-Qwen-Backend"))
        if mode == "drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.close_connection = True
            return
        if mode == "hold":
            self.server.hold_started.set()
            self.server.hold_release.wait(10)
        if mode == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.chunk(b'data: {"token":"first"}\n\n')
            self.server.sse_first.set()
            self.server.sse_release.wait(10)
            self.chunk(b'data: {"token":"second"}\n\n')
            self.server.sse_second.set()
            self.chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self.reply(200, {"worker": self.server.worker_name, "mode": mode})

    def chunk(self, data):
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


@unittest.skipUnless(IMAGE, "set QWEN_TEST_HAPROXY_IMAGE to run real Docker HAProxy tests")
class GatewayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMAGE != DEFAULT_IMAGE:
            raise ValueError("QWEN_TEST_HAPROXY_IMAGE must equal the reviewed digest-pinned DEFAULT_IMAGE")
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-gateway-integration-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        key = cls.root / "api-key"
        key.write_text(SECRET + "\n")
        os.chmod(key, 0o600)
        cls.worker = MockWorker()
        cls.addClassCleanup(cls.worker.server_close)
        threading.Thread(target=cls.worker.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.worker.shutdown)
        cls.port = unused_port()
        cls.config = {"api_key_file": str(key), "gateway_image": IMAGE,
                      "gateway_port": cls.port, "gateway_max_body_bytes": 256,
                      "gateway_backend_maxconn": 1, "gateway_backend_maxqueue": 1,
                      "gateway_timeout_queue_s": 2, "gateway_timeout_client_s": 15,
                      "gateway_timeout_server_s": 15, "gateway_ready_timeout_s": 15}
        cls.gateway = Gateway(cls.config, cls.root / "gateway")
        cls.addClassCleanup(cls.gateway.stop)
        cls.base_url = cls.gateway.start([cls.worker.url])

    def setUp(self):
        self.worker.hold_started.clear()
        self.worker.hold_release.clear()
        self.worker.sse_first.clear()
        self.worker.sse_release.clear()
        self.worker.sse_second.clear()
        self.addCleanup(self.worker.hold_release.set)
        self.addCleanup(self.worker.sse_release.set)

    def request(self, method="GET", path="/v1/models", body=None, token=SECRET, port=None,
                session=None, extra_headers=None, response_headers=False):
        connection = HTTPConnection("127.0.0.1", port or self.port, timeout=8)
        try:
            headers = {"Content-Type": "application/json"}
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            if session is not None:
                headers["X-Session-ID"] = session
            headers.update(extra_headers or {})
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            result = response.status, response.read()
            return (*result, dict(response.getheaders())) if response_headers else result
        finally:
            connection.close()

    @contextmanager
    def isolated_pool(self, **overrides):
        workers = [MockWorker(), MockWorker()]
        for worker in workers:
            threading.Thread(target=worker.serve_forever, daemon=True).start()
        port = unused_port()
        proxy = Gateway({**self.config, "gateway_port": port, **overrides}, self.root / "affinity")
        try:
            proxy.start([worker.url for worker in workers])
            seen = set()
            deadline = time.monotonic() + 6
            while len(seen) < 2 and time.monotonic() < deadline:
                status, body = self.request(port=port)
                if status == 200:
                    seen.add(json.loads(body)["worker"])
                time.sleep(0.05)
            self.assertEqual(seen, {worker.worker_name for worker in workers})
            yield workers, port
        finally:
            for worker in workers:
                worker.hold_release.set()
            proxy.stop()
            for worker in workers:
                worker.shutdown()
                worker.server_close()

    def test_authentication_missing_wrong_and_correct(self):
        before = len(self.worker.calls)
        self.assertEqual(self.request(token=None)[0], 401)
        self.assertEqual(self.request(token="wrong-token")[0], 401)
        self.assertEqual(len(self.worker.calls), before)
        status, body = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["worker"], self.worker.worker_name)

    def test_session_header_validation_and_backend_identity_cannot_be_spoofed(self):
        before = len(self.worker.calls)
        for session in ("", "a" * 129, "conversation with spaces", "a,b", "a/b", "a:b", "é"):
            with self.subTest(session=session):
                self.assertEqual(self.request(session=session)[0], 400)
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.putrequest("GET", "/v1/models")
            connection.putheader("Authorization", f"Bearer {SECRET}")
            connection.putheader("X-Session-ID", "conversation-a")
            connection.putheader("X-Session-ID", "conversation-b")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()
        self.assertEqual(len(self.worker.calls), before)
        for session in ("a", "a" * 128, "Session_0123.uuid-456"):
            status, _, headers = self.request(session=session, extra_headers={"X-Qwen-Backend": "client-spoof"},
                                              response_headers=True)
            self.assertEqual(status, 200)
            self.assertEqual(headers.get("x-qwen-backend"), "worker1")
            self.assertIsNone(self.worker.incoming_backend_headers[-1])

    def test_affinity_retains_busy_replica_and_bounds_its_queue(self):
        with self.isolated_pool() as (workers, port):
            status, body, headers = self.request(port=port, session="conversation-a", response_headers=True)
            self.assertEqual(status, 200)
            assigned = next(worker for worker in workers if worker.worker_name == json.loads(body)["worker"])
            other = next(worker for worker in workers if worker is not assigned)
            original_backend = headers.get("x-qwen-backend")
            self.assertIn(original_backend, ("worker1", "worker2"))
            with ThreadPoolExecutor(max_workers=2) as executor:
                active = executor.submit(self.request, "POST", "/v1/chat/completions", b'{"mode":"hold"}',
                                         SECRET, port, "conversation-a", response_headers=True)
                try:
                    self.assertTrue(assigned.hold_started.wait(5), "existing session moved to another replica")
                    queued = executor.submit(self.request, port=port, session="conversation-a", response_headers=True)
                    time.sleep(0.3)
                    self.assertFalse(queued.done(), "affinity was lost while its replica was busy")
                    started = time.monotonic()
                    self.assertEqual(self.request(port=port, session="conversation-a")[0], 503)
                    self.assertLess(time.monotonic() - started, 1.5)
                    # Both missing IDs and new conversations may use the idle worker.
                    for session in (None, "conversation-b"):
                        status, body, routed = self.request(port=port, session=session, response_headers=True)
                        self.assertEqual(status, 200)
                        self.assertEqual(json.loads(body)["worker"], other.worker_name)
                        self.assertNotEqual(routed.get("x-qwen-backend"), original_backend)
                finally:
                    assigned.hold_release.set()
                for completed in (active.result(timeout=5), queued.result(timeout=5)):
                    self.assertEqual(completed[0], 200)
                    self.assertEqual(completed[2].get("x-qwen-backend"), original_backend)

    def test_unhealthy_replica_reassigns_session_for_future_requests(self):
        with self.isolated_pool() as (workers, port):
            status, body, headers = self.request(port=port, session="conversation-failover", response_headers=True)
            self.assertEqual(status, 200)
            original = headers.get("x-qwen-backend")
            assigned = next(worker for worker in workers if worker.worker_name == json.loads(body)["worker"])
            assigned.healthy = False
            deadline = time.monotonic() + 9  # Three failed health probes at the configured 2s interval.
            replacement = original
            while replacement == original and time.monotonic() < deadline:
                status, _, response = self.request(port=port, session="conversation-failover", response_headers=True)
                self.assertEqual(status, 200)
                replacement = response.get("x-qwen-backend")
                time.sleep(0.1)
            self.assertIn(replacement, ("worker1", "worker2"))
            self.assertNotEqual(replacement, original)
            self.assertEqual(self.request(port=port, session="conversation-failover", response_headers=True)[2]
                             .get("x-qwen-backend"), replacement)

    def test_full_affinity_table_preserves_existing_sessions_and_allows_no_header(self):
        # Enable SC0 rate limiting alongside SC1 admission to catch counter/table mixups.
        with self.isolated_pool(gateway_session_table_size=2, gateway_rate_limit_rps=1000) as (workers, port):
            routes = {}
            for session in ("table-session-a", "table-session-b"):
                status, _, headers = self.request(port=port, session=session, response_headers=True)
                self.assertEqual(status, 200)
                routes[session] = headers.get("x-qwen-backend")
                self.assertIn(routes[session], ("worker1", "worker2"))
            before = sum(len(worker.calls) for worker in workers)
            self.assertEqual(self.request(port=port, session="table-session-c")[0], 503)
            self.assertEqual(sum(len(worker.calls) for worker in workers), before)
            for session, original_backend in routes.items():
                status, _, headers = self.request(port=port, session=session, response_headers=True)
                self.assertEqual(status, 200)
                self.assertEqual(headers.get("x-qwen-backend"), original_backend)
            self.assertEqual(self.request(port=port)[0], 200)

    def test_last_affinity_slot_is_reserved_while_request_waits_in_backend_queue(self):
        with self.isolated_pool(gateway_session_table_size=1) as (workers, port):
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Fill the workers without creating any affinity entries.
                active = [executor.submit(self.request, "POST", "/v1/chat/completions", b'{"mode":"hold"}', SECRET, port)
                          for _ in workers]
                try:
                    self.assertTrue(all(worker.hold_started.wait(5) for worker in workers))
                    pending = executor.submit(self.request, port=port, session="reserved-session")
                    time.sleep(0.3)
                    self.assertFalse(pending.done())
                    # The pending request has no backend connection yet. Its
                    # reservation must still prevent a second ID taking the slot.
                    started = time.monotonic()
                    self.assertEqual(self.request(port=port, session="overflow-session")[0], 503)
                    self.assertLess(time.monotonic() - started, 1.5)
                finally:
                    for worker in workers:
                        worker.hold_release.set()
                self.assertEqual(pending.result(timeout=5)[0], 200)
                for completed in active:
                    self.assertEqual(completed.result(timeout=5)[0], 200)
            self.assertEqual(self.request(port=port, session="reserved-session")[0], 200)

    def test_request_body_cap_and_chunked_upload_rejection(self):
        before = len(self.worker.calls)
        self.assertEqual(self.request("POST", "/v1/chat/completions", b"x" * 257)[0], 413)
        self.assertEqual(len(self.worker.calls), before)
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", body=iter([b'{}']),
                               headers={"Authorization": f"Bearer {SECRET}"}, encode_chunked=True)
            response = connection.getresponse()
            self.assertEqual(response.status, 411)
            response.read()
        finally:
            connection.close()
        self.assertEqual(len(self.worker.calls), before)

    def test_sse_is_delivered_before_generation_finishes(self):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", body=b'{"mode":"sse"}',
                               headers={"Authorization": f"Bearer {SECRET}"})
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
            self.assertEqual(response.readline(), b'data: {"token":"first"}\n')
            self.assertTrue(self.worker.sse_first.wait(1))
            self.assertFalse(self.worker.sse_second.is_set(), "gateway buffered until the second event")
            self.worker.sse_release.set()
            remaining = response.read()
            self.assertIn(b'"second"', remaining)
            self.assertIn(b"[DONE]", remaining)
        finally:
            self.worker.sse_release.set()
            connection.close()

    def test_bounded_queue_rejects_overload_and_expires_waiter(self):
        before = self.worker.calls.count("hold")
        with ThreadPoolExecutor(max_workers=2) as executor:
            active = executor.submit(self.request, "POST", "/v1/chat/completions", b'{"mode":"hold"}')
            self.assertTrue(self.worker.hold_started.wait(5))
            queued = executor.submit(self.request, "POST", "/v1/chat/completions", b'{"mode":"hold"}')
            try:
                time.sleep(0.3)  # Let the second request enter HAProxy's queue.
                started = time.monotonic()
                self.assertEqual(self.request("POST", "/v1/chat/completions", b'{"mode":"hold"}')[0], 503)
                self.assertLess(time.monotonic() - started, 1.5, "overload waited for the full queue timeout")
                self.assertEqual(queued.result(timeout=4)[0], 503)
                self.assertEqual(self.worker.calls.count("hold"), before + 1)
            finally:
                self.worker.hold_release.set()
            self.assertEqual(active.result(timeout=5)[0], 200)

    def test_failed_post_is_not_automatically_replayed(self):
        before = self.worker.calls.count("drop")
        self.assertEqual(self.request("POST", "/v1/chat/completions", b'{"mode":"drop"}')[0], 502)
        self.assertEqual(self.worker.calls.count("drop"), before + 1)

    def test_leastconn_uses_idle_worker_during_active_stream(self):
        other = MockWorker()
        other_thread = threading.Thread(target=other.serve_forever, daemon=True)
        other_thread.start()
        port = unused_port()
        proxy = Gateway({**self.config, "gateway_port": port}, self.root / "balanced")
        try:
            proxy.start([self.worker.url, other.url])
            # /health means at least one available worker; wait for both initial
            # health checks before testing distribution across the whole pool.
            seen = set()
            deadline = time.monotonic() + 6
            while len(seen) < 2 and time.monotonic() < deadline:
                status, body = self.request(port=port)
                if status == 200:
                    seen.add(json.loads(body)["worker"])
                time.sleep(0.05)
            self.assertEqual(seen, {self.worker.worker_name, other.worker_name})
            with ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(self.request, "POST", "/v1/chat/completions", b'{"mode":"hold"}', SECRET, port)
                try:
                    deadline = time.monotonic() + 5
                    while not self.worker.hold_started.is_set() and not other.hold_started.is_set():
                        if time.monotonic() >= deadline:
                            self.fail("neither backend received the active request")
                        time.sleep(0.02)
                    busy = self.worker if self.worker.hold_started.is_set() else other
                    idle = other if busy is self.worker else self.worker
                    status, body = self.request(port=port)
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["worker"], idle.worker_name)
                finally:
                    self.worker.hold_release.set()
                    other.hold_release.set()
                self.assertEqual(active.result(timeout=5)[0], 200)
        finally:
            proxy.stop()
            other.shutdown()
            other.server_close()


if __name__ == "__main__":
    unittest.main()
