"""CPU-only gateway tests. Docker and HTTP calls are always mocked."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_bench import gateway


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.key = self.directory / "api-key"
        self.key.write_text("test-secret-not-a-real-token\n")
        self.config = {"api_key_file": str(self.key)}
        self.backends = ["http://127.0.0.1:18100/v1", "http://localhost:18101"]
        self.instance = gateway.Gateway(self.config, self.directory / "run")
        self.container_id = "a" * 64
        self.calls = []
        self.launched = False

    def docker(self, args, **kwargs):
        self.calls.append(args)
        self.assertFalse(kwargs.get("shell", False))
        if args[:2] == ["docker", "run"]:
            self.assertEqual(args[args.index("--pull") + 1], "never")
            if "-c" in args:
                return subprocess.CompletedProcess(args, 0, "Configuration file is valid\n", "")
            self.launched = True
            return subprocess.CompletedProcess(args, 0, self.container_id + "\n", "")
        if args[:2] == ["docker", "inspect"]:
            if not self.launched or args[-1].endswith("-check"):
                return subprocess.CompletedProcess(args, 1, "", "Error: No such container")
            record = [{"Id": self.container_id, "Config": {"Labels": {
                gateway.OWNER_LABEL: self.instance.owner, gateway.ROLE_LABEL: gateway.ROLE}}}]
            return subprocess.CompletedProcess(args, 0, json.dumps(record), "")
        if args[:2] == ["docker", "rm"]:
            self.launched = False
            return subprocess.CompletedProcess(args, 0, self.container_id, "")
        self.fail(f"Unexpected Docker call: {args}")

    def healthy(self):
        response = MagicMock()
        response.__enter__.return_value.status = 200
        return response

    def test_export_enforces_auth_body_limit_and_streaming_admission(self):
        path = self.instance.export(self.backends, self.directory / "export")
        text = path.read_text()
        self.assertEqual(path.name, "haproxy.cfg")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn(self.key.read_text().strip(), text)
        digest = hashlib.sha256(b"Bearer test-secret-not-a-real-token").hexdigest()
        self.assertIn(digest, text)
        self.assertIn("return status 401 hdr WWW-Authenticate Bearer unless single_auth authorized", text)
        self.assertIn("req.hdr_cnt(transfer-encoding) gt 0", text)
        self.assertIn("return status 411 if body_method !has_length", text)
        self.assertIn("deny_status 413 if { req.hdr_val(content-length) gt 67108864 }", text)
        self.assertIn("bind 127.0.0.1:18080 proto h1", text)
        self.assertIn("maxconn 4 maxqueue 8", text)
        self.assertIn("queue(llm_pool) ge 16", text)
        self.assertIn("balance leastconn", text)
        self.assertIn("init-state fully-down", text)
        self.assertIn("retries 0\n    retry-on none", text)
        self.assertIn("timeout server 900s", text)
        self.assertIn("no option http-buffer-request", text)
        self.assertIn("option http-no-delay", text)
        self.assertNotIn("compression algo", text)
        self.assertNotIn("stick-table", text)
        self.assertIn("server worker2 127.0.0.1:18101 proto h1", text)

    def test_overrides_and_optional_rate_limit(self):
        config = {**self.config, "gateway_port": 8000, "gateway_backend_maxconn": 7,
                  "gateway_backend_maxqueue": 3, "gateway_timeout_server_s": 1200,
                  "gateway_rate_limit_rps": 20, "gateway_max_body_bytes": 12345}
        item = gateway.Gateway(config, self.directory / "run")
        text = item.export(["http://[::1]:18100/"], self.directory / "explicit.cfg").read_text()
        self.assertIn("bind 127.0.0.1:8000", text)
        self.assertIn("maxconn 7 maxqueue 3", text)
        self.assertIn("queue(llm_pool) ge 3", text)
        self.assertIn("sc_http_req_rate(0) gt 20", text)
        self.assertIn("timeout server 1200s", text)
        self.assertIn("content-length) gt 12345", text)
        self.assertIn("[::1]:18100", text)

    def test_rejects_unsafe_backend_urls_and_duplicates(self):
        bad = ["https://localhost:18100", "http://example.com:18100", "http://localhost",
               "http://localhost:0", "http://localhost:65536", "http://localhost:18100/private",
               "http://user:password@localhost:18100", "http://localhost:18100/?x=1",
               "http://localhost:18100/#x", "http://localhost:18100\nbackend injected",
               "http://localhost:18080"]
        for url in bad:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.instance.export([url], self.directory / "bad.cfg")
        for urls in ([], [self.backends[0], "http://localhost:18100"], "http://localhost:18100"):
            with self.subTest(urls=urls), self.assertRaises(ValueError):
                self.instance.export(urls, self.directory / "bad.cfg")
        self.assertFalse((self.directory / "bad.cfg").exists())

    def test_config_rejects_unbounded_or_injected_values(self):
        for name, value in [("gateway_port", 80), ("gateway_port", True), ("gateway_port", 65536),
                            ("gateway_maxconn", 0), ("gateway_backend_maxqueue", 0),
                            ("gateway_timeout_server_s", "900s\nmalicious"),
                            ("gateway_max_body_bytes", -1), ("gateway_rate_limit_rps", -1),
                            ("gateway_image", "--privileged"), ("gateway_image", "haproxy:latest"),
                            ("runtime_restart", "surprise")]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                gateway.Gateway({**self.config, name: value}, self.directory / "run").export(
                    self.backends, self.directory / "bad.cfg")

    def test_requires_one_nonempty_secret_and_refuses_symlinks(self):
        for key in ("", "first\nsecond", "has space", "unicode-é"):
            self.key.write_text(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.instance.export(self.backends, self.directory / "bad.cfg")
        with self.assertRaisesRegex(ValueError, "api_key_file"):
            gateway.Gateway({}, self.directory / "run").export(self.backends, self.directory / "bad.cfg")
        self.key.write_text("good-secret")
        output = self.directory / "output.cfg"
        output.symlink_to(self.key)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.instance.export(self.backends, output)
        self.assertEqual(self.key.read_text(), "good-secret")

    def test_start_validates_before_launch_and_records_ownership(self):
        with patch.object(gateway.subprocess, "run", side_effect=self.docker), \
                patch.object(gateway, "urlopen", return_value=self.healthy()) as http:
            self.assertEqual(self.instance.start(self.backends), "http://127.0.0.1:18080")
            record = json.loads((self.instance.run_dir / "gateway.json").read_text())
            self.assertEqual(record["status"], "running")
            self.assertEqual(record["owner"], self.instance.owner)
            self.assertEqual(record["container_id"], self.container_id)
            self.assertIn("-c", self.calls[0])
            self.assertIn("--detach", self.calls[1])
            self.assertIn(gateway.DEFAULT_IMAGE, self.calls[1])
            self.assertIn("--network", self.calls[1])
            self.assertEqual(self.calls[1][self.calls[1].index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
            http.assert_called_once_with("http://127.0.0.1:18080/health", timeout=2)
            with self.assertRaisesRegex(RuntimeError, "already been started"):
                self.instance.start(self.backends)
            self.instance.stop()
            self.instance.stop()  # Repeated stop is harmless and makes no extra Docker calls.
        removals = [args for args in self.calls if args[:2] == ["docker", "rm"]]
        self.assertEqual(removals, [["docker", "rm", "--force", self.container_id]])
        self.assertEqual(json.loads((self.instance.run_dir / "gateway.json").read_text())["status"], "stopped")

    def test_syntax_failure_never_launches_serving_container(self):
        def invalid(args, **kwargs):
            if args[:2] == ["docker", "run"]:
                self.calls.append(args)
                return subprocess.CompletedProcess(args, 1, "", "invalid syntax")
            return self.docker(args, **kwargs)
        with patch.object(gateway.subprocess, "run", side_effect=invalid), \
                patch.object(gateway, "urlopen") as http, \
                self.assertRaisesRegex(RuntimeError, "invalid syntax"):
            self.instance.start(self.backends)
        self.assertFalse(any("--detach" in args for args in self.calls))
        http.assert_not_called()

    def test_readiness_timeout_cleans_owned_container(self):
        self.config["gateway_ready_timeout_s"] = 1
        with patch.object(gateway.subprocess, "run", side_effect=self.docker), \
                patch.object(gateway.time, "monotonic", side_effect=[0, 0, 2]), \
                patch.object(gateway.time, "sleep"), \
                patch.object(gateway, "urlopen", side_effect=OSError("not ready")), \
                self.assertRaisesRegex(RuntimeError, "readiness timed out"):
            self.instance.start(self.backends)
        self.assertIn(["docker", "rm", "--force", self.container_id], self.calls)

    def test_crash_recovery_uses_manifest_and_matching_labels(self):
        with patch.object(gateway.subprocess, "run", side_effect=self.docker), \
                patch.object(gateway, "urlopen", return_value=self.healthy()):
            self.instance.start(self.backends)
            gateway.recover(self.instance.run_dir)
        self.assertIn(["docker", "inspect", "--type", "container", self.instance.name], self.calls)
        self.assertIn(["docker", "rm", "--force", self.container_id], self.calls)

    def test_uncertain_docker_launch_is_cleaned_by_owned_name(self):
        def launch_timeout(args, **kwargs):
            result = self.docker(args, **kwargs)
            if "--detach" in args:
                raise subprocess.TimeoutExpired(args, 60)
            return result
        with patch.object(gateway.subprocess, "run", side_effect=launch_timeout), \
                patch.object(gateway, "urlopen") as http, \
                self.assertRaises(subprocess.TimeoutExpired):
            self.instance.start(self.backends)
        self.assertIn(["docker", "inspect", "--type", "container", self.instance.name], self.calls)
        self.assertIn(["docker", "rm", "--force", self.container_id], self.calls)
        http.assert_not_called()

    def test_live_manifest_cannot_be_silently_overwritten(self):
        self.instance.run_dir.mkdir()
        manifest = self.instance.run_dir / "gateway.json"
        original = json.dumps({"status": "running", "owner": "old-owner"})
        manifest.write_text(original)
        with patch.object(gateway.subprocess, "run") as run, \
                self.assertRaisesRegex(RuntimeError, "recover this run"):
            self.instance.start(self.backends)
        self.assertEqual(manifest.read_text(), original)
        run.assert_not_called()

    def test_cleanup_refuses_wrong_owner_and_docker_daemon_failures(self):
        for owner, stderr in (("someone-else", ""), (None, "Cannot connect to Docker daemon")):
            inspect = [{"Id": self.container_id, "Config": {"Labels": {
                gateway.OWNER_LABEL: owner, gateway.ROLE_LABEL: gateway.ROLE}}}]
            response = subprocess.CompletedProcess([], int(owner is None), json.dumps(inspect), stderr)
            with self.subTest(owner=owner), patch.object(gateway.subprocess, "run", return_value=response) as run:
                with self.assertRaises(RuntimeError):
                    gateway._remove_owned(self.instance.name, self.instance.owner)
                self.assertEqual(run.call_count, 1)

    def test_recovery_rejects_forged_manifest_without_docker(self):
        self.instance.run_dir.mkdir()
        (self.instance.run_dir / "gateway.json").write_text(json.dumps({
            "schema": 1, "owner": self.instance.owner, "name": "unrelated-container"}))
        with patch.object(gateway.subprocess, "run") as run, self.assertRaises(ValueError):
            gateway.recover(self.instance.run_dir)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
