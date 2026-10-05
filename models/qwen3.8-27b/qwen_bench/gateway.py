"""HAProxy lifecycle for local vLLM workers; no Python request proxy.

HAProxy streams responses as they arrive. Requests must be stateless: callers
send the full conversation and use ``store: false`` with the Responses API.
The loopback endpoint deliberately requires an external TLS ingress for remote
access. Request bodies require Content-Length; chunked *responses* remain valid.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid


DEFAULT_IMAGE = (
    "haproxy:3.2.25-alpine@sha256:"
    "5d97434a423c2533cfeb42874d45cf6d69a840b60334582dfbfd5008e94c80be"
)
OWNER_LABEL = "io.glm-flash-a100.qwen-bench.owner"
ROLE_LABEL = "io.glm-flash-a100.qwen-bench.role"
ROLE = "gateway"
CONTAINER_CONFIG = "/usr/local/etc/haproxy/haproxy.cfg"
_OWNER = re.compile(r"^[a-f0-9]{32}$")
_CONTAINER_ID = re.compile(r"^[a-f0-9]{12,64}$")


def _get(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _integer(config, name, default, minimum=1, maximum=2_147_483_647):
    value = _get(config, name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def _safe_path(value):
    path = Path(value).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Refusing a symlink path: {path}")
    if any(char in str(path) for char in ("\n", "\r", "\x00", ",")):
        raise ValueError("Gateway paths cannot contain newlines, NUL, or commas")
    return path


def _write_private(path, text):
    path = _safe_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _command(args, *, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        # Commands never contain the API key; config diagnostics contain its digest only.
        detail = (result.stderr or result.stdout or "no diagnostics").strip()[-4000:]
        raise RuntimeError(f"{args[0]} {args[1]} failed ({result.returncode}): {detail}")
    return result.stdout.strip()


def _backend_address(url):
    if not isinstance(url, str) or any(char.isspace() for char in url):
        raise ValueError("Backend URLs must be HTTP loopback URLs with explicit ports")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid backend URL") from error
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ("", "/", "/v1", "/v1/")
            or port is None or not 1 <= port <= 65535):
        raise ValueError("Backend URLs must be HTTP loopback URLs with explicit ports and optional /v1")
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"{host}:{port}"


def _remove_owned(reference, owner):
    """Inspect then remove by immutable ID, never remove a reused container name."""
    result = subprocess.run(["docker", "inspect", "--type", "container", reference],
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        # A missing container is an idempotent cleanup; daemon failures are not.
        message = result.stderr.lower()
        if "no such object" in message or "no such container" in message:
            return False
        raise RuntimeError(f"Cannot inspect gateway container: {result.stderr.strip()[-1000:]}")
    try:
        records = json.loads(result.stdout)
        record = records[0] if len(records) == 1 else {}
        labels = record.get("Config", {}).get("Labels") or {}
        container_id = record.get("Id", "")
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise RuntimeError("Invalid Docker inspect response during gateway cleanup") from error
    if labels.get(OWNER_LABEL) != owner or labels.get(ROLE_LABEL) != ROLE:
        raise RuntimeError("Refusing to remove a gateway container with different ownership labels")
    if not _CONTAINER_ID.fullmatch(container_id):
        raise RuntimeError("Invalid gateway container ID returned by Docker")
    _command(["docker", "rm", "--force", container_id])
    return True


def recover(run_dir):
    """Clean a crashed run using its manifest and matching Docker ownership labels."""
    manifest = _safe_path(Path(run_dir) / "gateway.json")
    if not manifest.exists():
        return
    if not manifest.is_file() or manifest.stat().st_size > 65536:
        raise ValueError("Invalid gateway ownership manifest")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    owner = record.get("owner", "")
    name = record.get("name", "")
    if (record.get("schema") != 1 or not isinstance(owner, str) or not _OWNER.fullmatch(owner)
            or name != f"qwen-gateway-{owner}"):
        raise ValueError("Invalid gateway ownership manifest")
    for reference in (name, f"{name}-check"):
        _remove_owned(reference, owner)
    record["status"] = "stopped"
    _write_private(manifest, json.dumps(record, indent=2) + "\n")


class Gateway:
    """Manage one owner-labelled HAProxy container and a protected configuration."""

    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = _safe_path(run_dir)
        self.port = _integer(config, "gateway_port", 18080, maximum=65535)
        if self.port < 1024:
            raise ValueError("gateway_port must be an unprivileged port (1024 or higher)")
        self.image = _get(config, "gateway_image", DEFAULT_IMAGE)
        if (not isinstance(self.image, str) or not self.image
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./_:@-]*@sha256:[a-f0-9]{64}", self.image)):
            raise ValueError("gateway_image must have an immutable sha256 digest")
        self.restart = _get(config, "runtime_restart", _get(config, "restart_policy", "no"))
        if self.restart not in ("no", "unless-stopped", "always", "on-failure"):
            raise ValueError("runtime_restart must be a supported Docker restart policy")
        self.owner = uuid.uuid4().hex
        self.name = f"qwen-gateway-{self.owner}"
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.container_id = None
        self._attempted = False
        self._record = None

    def _render(self, backends):
        if not isinstance(backends, (list, tuple)) or not backends or len(backends) > 64:
            raise ValueError("Provide between one and 64 backend URLs")
        addresses = [_backend_address(url) for url in backends]
        if len(set(addresses)) != len(addresses):
            raise ValueError("Backend URLs must be unique")
        if any(address in (f"127.0.0.1:{self.port}", f"[::1]:{self.port}") for address in addresses):
            raise ValueError("The gateway cannot be its own backend")

        maximum = _integer(self.config, "gateway_maxconn", 64, maximum=65536)
        active = _integer(self.config, "gateway_backend_maxconn", 4, maximum=65536)
        queued = _integer(self.config, "gateway_backend_maxqueue", 8, maximum=65536)
        body = _integer(self.config, "gateway_max_body_bytes", 64 * 1024 * 1024)
        rate = _integer(self.config, "gateway_rate_limit_rps", 0, minimum=0, maximum=1000000)
        timeouts = {key: _integer(self.config, f"gateway_timeout_{key}_s", default, maximum=86400)
                    for key, default in (("connect", 5), ("queue", 30), ("client", 900), ("server", 900))}
        key_file = _get(self.config, "api_key_file")
        if not key_file:
            raise ValueError("api_key_file is required for gateway authentication")
        key_path = _safe_path(key_file)
        if not key_path.is_file() or key_path.stat().st_size > 4096:
            raise ValueError("api_key_file must contain one API key of at most 4096 bytes")
        key = key_path.read_text(encoding="utf-8").strip()
        if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise ValueError("api_key_file must contain a single printable ASCII token without whitespace")
        digest = hashlib.sha256(f"Bearer {key}".encode("ascii")).hexdigest()

        lines = [
            "# Generated HAProxy configuration. API keys are represented only by a SHA-256 digest.",
            "# Stateless full-context requests only; Responses callers must use store:false.",
            "global",
            "    log stdout format raw local0",
            "    nbthread 1",  # Small stream counts; serial admission keeps the queue check deterministic.
            f"    maxconn {maximum}",
            "",
            "defaults",
            "    log global",
            "    mode http",
            "    option httplog",
            "    option dontlognull",
            "    option http-no-delay",
            "    option abortonclose",
            "    no option http-buffer-request",
            "    retries 0",
            "    retry-on none",
            "    timeout http-request 30s",
            "    timeout http-keep-alive 15s",
        ]
        lines.extend(f"    timeout {key} {value}s" for key, value in timeouts.items())
        lines += [
            "",
            "frontend llm_frontend",
            f"    bind 127.0.0.1:{self.port} proto h1",
            f"    maxconn {maximum}",
            "    monitor-uri /health",
            "    monitor fail if { nbsrv(llm_pool) lt 1 }",
            "    acl single_auth req.fhdr_cnt(authorization) eq 1",
            f"    acl authorized req.fhdr(authorization),sha2(256),hex -m str -i {digest}",
            "    http-request return status 401 hdr WWW-Authenticate Bearer unless single_auth authorized",
            "    # Known length makes the body cap enforceable without buffering uploads.",
            "    http-request return status 411 if { req.hdr_cnt(transfer-encoding) gt 0 }",
            "    acl body_method method POST PUT PATCH",
            "    acl has_length req.hdr_cnt(content-length) eq 1",
            "    http-request return status 411 if body_method !has_length",
            f"    http-request deny deny_status 413 if {{ req.hdr_val(content-length) gt {body} }}",
            f"    http-request deny deny_status 503 if {{ queue(llm_pool) ge {queued * len(addresses)} }}",
        ]
        if rate:
            lines += [
                "    stick-table type integer size 1 expire 10s store http_req_rate(1s)",
                "    http-request track-sc0 int(1)",
                f"    http-request deny deny_status 429 if {{ sc_http_req_rate(0) gt {rate} }}",
            ]
        lines += [
            "    default_backend llm_pool",
            "",
            "backend llm_pool",
            "    balance leastconn",
            "    http-reuse safe",
            "    option httpchk",
            "    http-check send meth GET uri /health ver HTTP/1.1 hdr Host localhost",
            "    http-check expect status 200",
            f"    default-server check inter 2s fall 3 rise 1 init-state fully-down maxconn {active} maxqueue {queued}",
        ]
        lines.extend(f"    server worker{index} {address} proto h1" for index, address in enumerate(addresses, 1))
        return "\n".join(lines) + "\n"

    def export(self, backends, destination):
        """Write a mode-0600 config to a directory (or explicit .cfg filename)."""
        rendered = self._render(backends)
        path = _safe_path(destination)
        if path.suffix != ".cfg" or path.is_dir():
            path /= "haproxy.cfg"
        _write_private(path, rendered)
        return path

    def _docker_base(self, name, config_path):
        return ["docker", "run", "--pull", "never", "--name", name, "--network", "host",
                "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--log-driver", "json-file", "--log-opt", "max-size=10m", "--log-opt", "max-file=3",
                "--label", f"{OWNER_LABEL}={self.owner}", "--label", f"{ROLE_LABEL}={ROLE}",
                "--mount", f"type=bind,src={config_path},dst={CONTAINER_CONFIG},readonly"]

    def _save(self):
        _write_private(self.run_dir / "gateway.json", json.dumps(self._record, indent=2) + "\n")

    def start(self, backends):
        """Validate using the already-pulled image, launch, then await worker health."""
        if self._attempted:
            raise RuntimeError("This Gateway has already been started; stop it before starting again")
        ready_timeout = _integer(self.config, "gateway_ready_timeout_s", 30, maximum=86400)
        manifest = _safe_path(self.run_dir / "gateway.json")
        if manifest.exists():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous.get("status") != "stopped":
                raise RuntimeError("A gateway manifest already exists; recover this run before restarting")
        config_path = self.export(backends, self.run_dir)
        self._record = {"schema": 1, "owner": self.owner, "name": self.name,
                        "container_id": None, "status": "starting", "image": self.image,
                        "base_url": self.base_url, "backends": list(backends), "config": str(config_path)}
        self._save()
        self._attempted = True
        try:
            _command(self._docker_base(f"{self.name}-check", config_path)
                     + ["--rm", self.image, "haproxy", "-c", "-f", CONTAINER_CONFIG], timeout=60)
            container_id = _command(self._docker_base(self.name, config_path)
                                    + ["--detach", "--restart", self.restart,
                                       self.image, "haproxy", "-W", "-db", "-f", CONTAINER_CONFIG], timeout=60)
            if not _CONTAINER_ID.fullmatch(container_id):
                raise RuntimeError("Docker returned an invalid gateway container ID")
            self.container_id = container_id
            self._record.update(container_id=container_id, status="starting")
            self._save()
            deadline = time.monotonic() + ready_timeout
            last_error = "no health response"
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"{self.base_url}/health", timeout=min(2, ready_timeout)) as response:
                        if response.status == 200:
                            self._record["status"] = "running"
                            self._save()
                            return self.base_url
                        last_error = f"HTTP {response.status}"
                except (URLError, HTTPError, TimeoutError, OSError) as error:
                    last_error = str(error)
                time.sleep(0.2)
            raise RuntimeError(f"Gateway readiness timed out after {ready_timeout}s: {last_error}")
        except BaseException as error:
            try:
                self.stop()
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Gateway cleanup also failed; run recover: {cleanup_error}")
            raise

    def stop(self):
        """Remove only containers carrying this object's ownership and role labels."""
        if not self._attempted:
            return
        for reference in (self.container_id or self.name, f"{self.name}-check"):
            _remove_owned(reference, self.owner)
        self.container_id = None
        self._attempted = False
        if self._record is not None:
            self._record["status"] = "stopped"
            self._save()
