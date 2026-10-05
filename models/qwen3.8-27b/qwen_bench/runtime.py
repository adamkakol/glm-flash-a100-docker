"""Isolated Docker lifecycle for the A100 benchmark; never downloads implicitly.

All host code uses the Python standard library. Source/configuration support is
not a claim that any candidate passed the target GPU workload.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
from typing import Callable
import urllib.error
import urllib.request
import uuid

from .state import atomic_json
from .ports import check_ports, validate_port_layout


OWNER_LABEL = "io.qwen-bench.run"
MODEL_NAME = "qwen3.8-27b"
_REVISION = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}")
_GPU_PROBE = """
import importlib.metadata, json, subprocess
import torch
from packaging.version import Version
transformers_version = importlib.metadata.version('transformers')
assert Version(transformers_version) >= Version('5.8.0'), 'Transformers >=5.8.0 required'
assert torch.cuda.is_available(), 'Container CUDA is unavailable; check host driver compatibility'
assert torch.cuda.device_count() == 2, 'Expected exactly two selected NVLink GPUs'
capabilities = [torch.cuda.get_device_capability(i) for i in range(2)]
assert capabilities == [(8, 0), (8, 0)], f'Expected A100 SM80, got {capabilities}'
peers = [torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0)]
assert all(peers), 'Bidirectional CUDA peer access is unavailable'
for i in range(2):
    x = torch.ones((16, 16), device=f'cuda:{i}', dtype=torch.bfloat16)
    assert (x @ x).sum().item() == 4096, 'CUDA BF16 execution failed'
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
drivers = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                  '--format=csv,noheader'], text=True).splitlines()
print('QWEN_GPU_PROBE=' + json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'driver_versions': drivers, 'transformers': transformers_version,
    'vllm': importlib.metadata.version('vllm'), 'capabilities': capabilities,
    'peer_access': peers, 'qwen_model_import': True}))
"""


@dataclass(frozen=True)
class Candidate:
    id: str
    precision: str
    layout: str
    chunk_size: int
    draft_tokens: int = 0


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_mib: int
    used_mib: int
    utilization: int
    mig_current: str
    mig_pending: str


@dataclass(frozen=True)
class Topology:
    gpus: tuple[GPU, ...]
    nvlink_pair: tuple[str, str]
    topology_text: str

    def to_dict(self) -> dict:
        return asdict(self)


class TransientGPUBusy(RuntimeError):
    """Busy inventory reading with no selected GPU compute processes."""


def default_config() -> dict:
    """Return editable defaults with reviewed immutable image/model pins."""
    return {
        "docker_image": "vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90",
        "models": {
            "bf16": {"repo_id": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"},
            "fp8": {"repo_id": "Qwen/Qwen3.8-27B-FP8", "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"},
        },
        "cache_dir": str(Path.home() / ".cache" / "qwen-bench"),
        "gpu_uuids": [], "base_port": 18100, "api_key_file": None,
        "max_model_len": 327680, "target_input_tokens": 260000,
        "output_tokens": 32768, "max_num_seqs": 4,
        "gpu_memory_utilization": 0.90, "kv_cache_dtype": "bfloat16",
        "chunks": [2048, 8192], "precisions": ["bf16", "fp8"],
        "layouts": ["tp2", "replicas"], "draft_tokens": [0, 1, 3],
        "enable_mtp": True, "startup_timeout_s": 1800,
        "image_limit": 4, "video_limit": 1, "video_num_frames": 32,
        "busy_memory_mib": 1024, "busy_utilization_percent": 5,
        "log_tail_lines": 2000, "log_max_bytes": 262144,
        "restart_policy": "no",
    }


def validate_config(config: dict) -> None:
    """Validate configuration without hardware access or filesystem writes."""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    for key in default_config():
        if key not in config:
            raise ValueError(f"missing configuration key: {key}")
    positive = ("max_model_len", "target_input_tokens", "output_tokens",
                "max_num_seqs", "startup_timeout_s", "image_limit", "video_limit",
                "video_num_frames", "log_tail_lines", "log_max_bytes")
    for key in positive:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 292768 <= config["max_model_len"] <= 524288:
        raise ValueError("max_model_len must be 292768..524288 for the factor-2 YaRN profile")
    if config["target_input_tokens"] + config["output_tokens"] > config["max_model_len"]:
        raise ValueError("input plus output exceeds max_model_len")
    if config["target_input_tokens"] < 260000 or config["max_num_seqs"] < 3:
        raise ValueError("retain the 260k target and at least three scheduler slots")
    validate_port_layout(config)
    if not isinstance(config["gpu_memory_utilization"], (int, float)) or not 0.5 <= config["gpu_memory_utilization"] <= 0.95:
        raise ValueError("gpu_memory_utilization must be between 0.5 and 0.95")
    if config["kv_cache_dtype"] != "bfloat16":
        raise ValueError("this A100 benchmark requires BF16 KV cache")
    for key, allowed in (("precisions", {"bf16", "fp8"}),
                         ("layouts", {"tp2", "replicas"}),
                         ("draft_tokens", {0, 1, 3})):
        values = config[key]
        if not isinstance(values, list) or not values or any(v not in allowed for v in values) or len(values) != len(set(values)):
            raise ValueError(f"invalid {key}")
    if 0 not in config["draft_tokens"]:
        raise ValueError("draft_tokens must include the non-speculative baseline 0")
    if not isinstance(config["chunks"], list) or not config["chunks"] or any(type(v) is not int or v < 512 or v % 256 for v in config["chunks"]):
        raise ValueError("chunks must be positive multiples of 256, at least 512")
    if type(config["enable_mtp"]) is not bool:
        raise ValueError("enable_mtp must be boolean")
    if config.get("reasoning_effort", "xhigh") not in {"low", "medium", "xhigh"}:
        raise ValueError("reasoning_effort must be low, medium, or xhigh")
    gpu_ids = config["gpu_uuids"]
    if not isinstance(gpu_ids, list) or (gpu_ids and (len(gpu_ids) != 3 or len(set(gpu_ids)) != 3 or any(not re.fullmatch(r"GPU-[a-zA-Z0-9-]+", v) for v in gpu_ids))):
        raise ValueError("gpu_uuids must be empty or contain three distinct full GPU UUIDs")
    for precision in config["precisions"]:
        model = config["models"].get(precision, {})
        expected = "Qwen/Qwen3.8-27B" + ("-FP8" if precision == "fp8" else "")
        if model.get("repo_id") != expected:
            raise ValueError(f"{precision} must use the official checkpoint {expected}")
        if model.get("revision") and not _REVISION.fullmatch(model["revision"]):
            raise ValueError("model revisions must be immutable 40-character commit hashes")
    if not isinstance(config["docker_image"], str) or any(c.isspace() for c in config["docker_image"]):
        raise ValueError("invalid docker_image")
    if config.get("runtime_restart", config["restart_policy"]) not in {"no", "unless-stopped"}:
        raise ValueError("restart_policy must be no or unless-stopped")
    for key in ("busy_memory_mib", "busy_utilization_percent"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"invalid {key}")
    if not isinstance(config["cache_dir"], str) or not config["cache_dir"]:
        raise ValueError("cache_dir must be a path")


def generate_candidates(config: dict) -> list[Candidate]:
    validate_config(config)
    drafts = config["draft_tokens"] if config["enable_mtp"] else [0]
    return [Candidate(f"{p}-{layout}-c{chunk}-mtp{draft}", p, layout, chunk, draft)
            for p, layout, chunk, draft in itertools.product(
                config["precisions"], config["layouts"], config["chunks"], drafts)]


def _run(command: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def _checked(runner: Callable, command: list[str], timeout: float = 30) -> str:
    result = runner(command, timeout=timeout)
    if result.returncode:
        # Commands never contain credentials. Avoid dumping arbitrary Docker output.
        raise RuntimeError(f"{command[0]} {command[1]} failed (exit {result.returncode})")
    return result.stdout


def _remove(runner: Callable, name: str) -> None:
    result = runner(["docker", "rm", name], timeout=30)
    if result.returncode and "No such" not in result.stderr:
        raise RuntimeError("Docker container removal failed")


def parse_nvlink_pairs(text: str) -> list[tuple[int, int]]:
    """Parse both directions, preserving physical indices (not row order)."""
    header, rows = [], {}
    for line in text.splitlines():
        cells = line.split()
        if not cells:
            continue
        if not header and re.fullmatch(r"GPU\d+", cells[0]) and len(cells) > 1 and re.fullmatch(r"GPU\d+", cells[1]):
            header = [int(c[3:]) for c in cells if re.fullmatch(r"GPU\d+", c)]
        elif header and re.fullmatch(r"GPU\d+", cells[0]) and len(cells) > len(header):
            rows[int(cells[0][3:])] = dict(zip(header, cells[1:1 + len(header)]))
    linked = lambda value: bool(re.fullmatch(r"NV[1-9]\d*", value))
    return [(a, b) for a, b in itertools.combinations(sorted(rows), 2)
            if linked(rows[a].get(b, "")) and linked(rows[b].get(a, ""))]


def detect_gpus(config: dict | None = None, runner: Callable | None = None) -> Topology:
    config, runner = config or default_config(), runner or _run
    query = "index,uuid,name,memory.total,memory.used,utilization.gpu,mig.mode.current,mig.mode.pending"
    inventory = _checked(runner, ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    gpus = []
    try:
        for row in csv.reader(inventory.splitlines(), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 8:
                raise ValueError("unexpected GPU inventory columns")
            gpus.append(GPU(int(row[0]), row[1].strip(), row[2].strip(),
                            int(row[3]), int(row[4]), int(row[5]), row[6].strip(), row[7].strip()))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("unable to parse complete GPU inventory; fail closed") from exc
    requested = config["gpu_uuids"]
    if requested:
        inventory_by_uuid = {g.uuid: g for g in gpus}
        if any(g not in inventory_by_uuid for g in requested):
            raise RuntimeError("a selected GPU UUID is absent")
        gpus = [inventory_by_uuid[g] for g in requested]
    if len(gpus) != 3 or len({g.uuid for g in gpus}) != 3 or len({g.index for g in gpus}) != 3:
        raise RuntimeError("select exactly three distinct A100 80GB GPUs using gpu_uuids")
    for gpu in gpus:
        if "A100" not in gpu.name or not 75000 <= gpu.memory_mib <= 85000:
            raise RuntimeError(f"GPU {gpu.uuid} is not an A100 80GB")
        if gpu.mig_current != "Disabled" or gpu.mig_pending != "Disabled":
            raise RuntimeError(f"MIG must be disabled, including pending mode, for {gpu.uuid}")
    processes = _checked(runner, ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    if any(row and row[0].strip() in {g.uuid for g in gpus}
           for row in csv.reader(processes.splitlines(), skipinitialspace=True)):
        raise RuntimeError("selected GPUs have active compute processes")
    for gpu in gpus:
        if gpu.used_mib > config["busy_memory_mib"] or gpu.utilization > config["busy_utilization_percent"]:
            raise TransientGPUBusy(f"GPU {gpu.uuid} still reports busy memory/utilization without a compute process")
    topo = _checked(runner, ["nvidia-smi", "topo", "-m"])
    by_index = {g.index: g.uuid for g in gpus}
    pairs = [(by_index[a], by_index[b]) for a, b in parse_nvlink_pairs(topo)
             if a in by_index and b in by_index]
    if len(pairs) != 1:
        raise RuntimeError("expected exactly one bidirectional NVLink pair among the three selected GPUs")
    return Topology(tuple(gpus), pairs[0], topo)


def _detect_quiescent_gpus(config: dict, runner: Callable) -> Topology:
    """Allow a trailing nvidia-smi sample to settle; never wait on a process."""
    deadline = time.monotonic() + 10
    while True:
        try:
            return detect_gpus(config, runner)
        except TransientGPUBusy:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(1, remaining))


def recover(run_dir: str | Path, runner: Callable | None = None) -> None:
    """Clean a crashed run only after verifying each persisted ownership label."""
    runner = runner or _run
    manifest = Path(run_dir) / "runtime.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text())
    run_id = data.get("run_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise RuntimeError("invalid recovery run ID")
    remaining, errors = [], []
    for container in data.get("containers", []):
        name = container.get("name", "")
        try:
            if not re.fullmatch(r"qwen-bench-" + run_id + r"-(\d+|probe|download-(bf16|fp8))", name):
                raise RuntimeError("invalid recovery container name")
            result = runner(["docker", "inspect", "--format", '{{index .Config.Labels "' + OWNER_LABEL + '"}}', name], timeout=20)
            if result.returncode:
                if "No such" in result.stderr:
                    continue
                raise RuntimeError("Docker unavailable during recovery")
            if result.stdout.strip() != run_id:
                raise RuntimeError("recovery ownership label mismatch")
            _checked(runner, ["docker", "stop", "--time", "20", name], timeout=30)
            _remove(runner, name)
        except Exception as exc:
            remaining.append(container)
            errors.append(str(exc))
    data["containers"] = remaining
    atomic_json(manifest, data)
    (Path(run_dir) / f"runtime-{run_id}.env").unlink(missing_ok=True)
    if errors:
        raise RuntimeError("recovery incomplete: " + "; ".join(errors))


class DockerRuntime:
    def __init__(self, config: dict, run_dir: str | Path, *, runner: Callable | None = None,
                 topology: Topology | None = None):
        validate_config(config)
        self.config = json.loads(json.dumps(config))
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.run_dir / "runtime.json"
        if manifest.exists() and json.loads(manifest.read_text()).get("containers"):
            raise RuntimeError("run directory still records containers; call recover(run_dir) first")
        self.runner = runner or _run
        self.topology = topology
        self.run_id = uuid.uuid4().hex
        self.containers: list[dict] = []
        self._capacities: dict[str, int] = {}
        self.api_key = self._load_key()
        self.env_file = self.run_dir / f"runtime-{self.run_id}.env"
        self._candidate: Candidate | None = None

    def _persist(self) -> None:
        manifest = self.run_dir / "runtime.json"
        atomic_json(manifest, {"run_id": self.run_id,
                             "containers": self.containers,
                             "candidate": asdict(self._candidate) if self._candidate else None,
                             "image": self.config["docker_image"]})

    @staticmethod
    def _private_write(path: Path, value: str) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(value)

    def _load_key(self) -> str:
        supplied = self.config["api_key_file"]
        path = Path(supplied).expanduser().resolve() if supplied else self.run_dir / "api-key"
        if not path.exists():
            if supplied:
                raise ValueError("api_key_file does not exist")
            self._private_write(path, secrets.token_urlsafe(32) + "\n")
        if path.stat().st_mode & 0o077:
            raise ValueError("api_key_file must be private (chmod 600)")
        key = path.read_text().strip()
        if len(key) < 16 or any(c.isspace() for c in key):
            raise ValueError("API key must contain at least 16 non-whitespace characters")
        return key

    def _pins(self, precision: str) -> dict:
        if not _DIGEST.fullmatch(self.config["docker_image"]):
            raise ValueError("set docker_image to the reviewed image@sha256 digest before running")
        model = self.config["models"][precision]
        if not _REVISION.fullmatch(model["revision"]):
            raise ValueError(f"pin the {precision} model revision before running")
        return model

    def _snapshot_path(self, precision: str) -> Path:
        model = self._pins(precision)
        root = Path(self.config["cache_dir"]).expanduser().resolve()
        path = root / "hub" / ("models--" + model["repo_id"].replace("/", "--")) / "snapshots" / model["revision"]
        if not (path / "config.json").is_file():
            raise RuntimeError(f"{precision} snapshot missing; run the explicit download command first")
        indices = list(path.glob("*.safetensors.index.json"))
        if len(indices) != 1:
            raise RuntimeError("expected one safetensors index in the pinned snapshot")
        weight_map = json.loads(indices[0].read_text()).get("weight_map", {})
        shards = set(weight_map.values())
        if not shards or any(not isinstance(s, str) or Path(s).name != s or not (path / s).is_file() or (path / s).stat().st_size == 0 for s in shards):
            raise RuntimeError("pinned model snapshot is incomplete; no automatic download is allowed")
        if not (path / "tokenizer_config.json").is_file():
            raise RuntimeError("pinned model tokenizer is missing")
        return path

    def _vllm_args(self, candidate: Candidate, model_path: str, tp: int) -> list[str]:
        cfg = self.config
        rope = {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                "rope_type": "yarn", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25, "factor": 2.0,
                "original_max_position_embeddings": 262144}
        args = ["serve", model_path, "--served-model-name", MODEL_NAME,
                "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16",
                # v0.30 reserves explicit bfloat16 for models overriding FP8 KV;
                # auto follows the explicitly selected BF16 model compute dtype.
                "--kv-cache-dtype", "auto", "--tensor-parallel-size", str(tp),
                "--max-model-len", str(cfg["max_model_len"]),
                "--max-num-seqs", str(cfg["max_num_seqs"]),
                "--max-num-batched-tokens", str(candidate.chunk_size),
                "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
                "--enable-chunked-prefill", "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
                "--mamba-cache-mode", "align", "--reasoning-parser", "qwen3",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
                "--default-chat-template-kwargs", json.dumps({"reasoning_effort": cfg.get("reasoning_effort", "xhigh")}),
                "--hf-overrides", json.dumps({"text_config": {"rope_parameters": rope}}),
                "--limit-mm-per-prompt", json.dumps({"image": cfg["image_limit"], "video": cfg["video_limit"]}),
                "--media-io-kwargs", json.dumps({"video": {"num_frames": cfg["video_num_frames"]}}),
                "--no-enable-log-requests", "--sse-keep-alive-interval", "15",
                "--max-num-queued-reqs", str(2 * cfg["max_num_seqs"])]
        if candidate.precision == "fp8":
            args += ["--quantization", "fp8"]
        if candidate.draft_tokens:
            args += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": candidate.draft_tokens})]
        return args

    def launch_commands(self, candidate: Candidate) -> list[list[str]]:
        """Return secret-free argv lists; requires prepared files and topology."""
        if candidate not in generate_candidates(self.config):
            raise ValueError("candidate is outside the configured, supported matrix")
        if self.topology is None:
            raise RuntimeError("GPU topology must be detected before building launch commands")
        model_path = self._snapshot_path(candidate.precision)
        if not self.env_file.exists():
            self._private_write(self.env_file, f"VLLM_API_KEY={self.api_key}\n")
        cache = Path(self.config["cache_dir"]).expanduser().resolve()
        if any(c in str(cache) for c in (",", "\n")):
            raise ValueError("cache path must not contain commas or newlines")
        container_path = "/hf/" + str(model_path.relative_to(cache))
        groups = [list(self.topology.nvlink_pair)] if candidate.layout == "tp2" else [[g.uuid] for g in self.topology.gpus]
        commands = []
        for index, group in enumerate(groups):
            name = f"qwen-bench-{self.run_id}-{index}"
            # Docker's --gpus grammar requires quotes around a comma-separated device list.
            devices = '"device=' + ",".join(group) + '"'
            command = ["docker", "run", "--detach", "--pull=never", "--name", name,
                       "--label", f"{OWNER_LABEL}={self.run_id}",
                       "--label", f"io.qwen-bench.candidate={candidate.id}",
                       "--restart", self.config.get("runtime_restart", self.config["restart_policy"]),
                       "--gpus", devices, "--ipc", "private", "--shm-size", "16g",
                       "--publish", f"127.0.0.1:{self.config['base_port'] + index}:8000",
                       "--mount", f"type=bind,src={cache},dst=/hf,readonly",
                       "--env-file", str(self.env_file),
                       "--env", "HF_HOME=/hf", "--env", "HF_HUB_OFFLINE=1",
                       "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
                       "--env", "HF_HUB_DISABLE_TELEMETRY=1", "--env", "DO_NOT_TRACK=1",
                       "--env", "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
                       "--env", "VLLM_USE_DEEP_GEMM=0", "--env", "NCCL_DEBUG=INFO",
                       "--log-driver", "json-file", "--log-opt", "max-size=16m",
                       "--log-opt", "max-file=2", "--entrypoint", "vllm",
                       self.config["docker_image"]]
            commands.append(command + self._vllm_args(candidate, container_path, len(group)))
        return commands

    def _inspect(self, name: str) -> dict | None:
        template = '{{json .State}}\n{{index .Config.Labels "' + OWNER_LABEL + '"}}'
        result = self.runner(["docker", "inspect", "--format", template, name], timeout=20)
        if result.returncode:
            if "No such" in result.stderr:
                return None
            raise RuntimeError("Docker container inspection failed")
        lines = result.stdout.strip().splitlines()
        if len(lines) != 2 or lines[1] != self.run_id:
            raise RuntimeError("refusing to operate on a container without this run's ownership label")
        return json.loads(lines[0])

    def _capture_logs(self, name: str) -> str:
        result = self.runner(["docker", "logs", "--tail", str(self.config["log_tail_lines"]), name], timeout=20)
        tail = (result.stdout + result.stderr).replace(self.api_key, "[REDACTED]")
        capacities = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", tail)
        if capacities:
            self._capacities[name] = int(capacities[-1].replace(",", ""))
        tail = tail.encode("utf-8")[-self.config["log_max_bytes"]:].decode("utf-8", errors="replace")
        path = self.run_dir / (name + ".log")
        path.write_text(tail)
        return tail

    def _healthy(self, url: str) -> bool:
        request = urllib.request.Request(url + "/health", headers={"Authorization": "Bearer " + self.api_key})
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def start(self, candidate: Candidate) -> list[str]:
        if self.containers:
            raise RuntimeError("stop this runtime's previous candidate before starting another")
        if candidate not in generate_candidates(self.config):
            raise ValueError("candidate is outside the configured, supported matrix")
        check_ports(self.config, backend_count=1 if candidate.layout == "tp2" else 3, runner=self.runner)
        # Always refresh before claiming an idle device, even after a previous detection.
        detected = _detect_quiescent_gpus(self.config, self.runner)
        if self.topology and detected.nvlink_pair != self.topology.nvlink_pair:
            raise RuntimeError("NVLink GPU mapping changed since preflight")
        self.topology = detected
        commands = self.launch_commands(candidate)
        self._candidate = candidate
        self._capacities.clear()
        try:
            for index, command in enumerate(commands):
                name = command[command.index("--name") + 1]
                self.containers.append({"name": name, "url": f"http://127.0.0.1:{self.config['base_port'] + index}"})
                self._persist()
                result = self.runner(command, timeout=60)
                if result.returncode:
                    detail = (result.stderr or result.stdout).replace(self.api_key, "[REDACTED]")[-1600:].strip()
                    raise RuntimeError(f"Docker could not start backend {index} on "
                                       f"127.0.0.1:{self.config['base_port'] + index}: "
                                       f"{detail or ('exit ' + str(result.returncode))}")
            # Docker has already stored the environment; restart needs no host copy.
            self.env_file.unlink(missing_ok=True)
            deadline = time.monotonic() + self.config["startup_timeout_s"]
            while time.monotonic() < deadline:
                ready = True
                for container in self.containers:
                    state = self._inspect(container["name"])
                    self._capture_logs(container["name"])
                    if not state or not state.get("Running"):
                        raise RuntimeError(f"{container['name']} exited during startup; inspect its run log")
                    ready = self._healthy(container["url"]) and ready
                if ready:
                    return [c["url"] for c in self.containers]
                time.sleep(2)
            raise TimeoutError("vLLM startup timed out; inspect the bounded logs in the run directory")
        except BaseException:
            self.stop()
            raise

    def snapshot(self) -> dict:
        containers = []
        for container in self.containers:
            state = self._inspect(container["name"])
            tail = self._capture_logs(container["name"])
            containers.append({**container, "state": state,
                               "kv_capacity_tokens": self._capacities.get(container["name"]),
                               "log_file": str(self.run_dir / (container["name"] + ".log"))})
        return {"run_id": self.run_id, "candidate": asdict(self._candidate) if self._candidate else None,
                "topology": self.topology.to_dict() if self.topology else None,
                "containers": containers,
                "required_long_tokens": 2 * (self.config["target_input_tokens"] + self.config["output_tokens"])}

    def capacity(self) -> dict:
        """Observed hybrid KV capacity, never an estimate from free VRAM."""
        for container in self.containers:
            self._capture_logs(container["name"])
        required = self.config["max_model_len"] * (2 if self._candidate and self._candidate.layout == "tp2" else 1)
        observed = [self._capacities.get(c["name"]) for c in self.containers]
        errors = [f"backend {i}: KV capacity unknown" if n is None else
                  f"backend {i}: {n} KV tokens below required {required}"
                  for i, n in enumerate(observed) if n is None or n < required]
        if not observed:
            errors.append("no running candidate")
        elif self._candidate and len(observed) != (1 if self._candidate.layout == "tp2" else 3):
            errors.append("incomplete backend group")
        return {"passed": not errors, "engine_cache_tokens": observed,
                "required_per_engine": required, "errors": errors}

    def stop(self) -> None:
        errors, remaining = [], []
        for container in reversed(self.containers):
            name = container["name"]
            try:
                state = self._inspect(name)
                if state is None:
                    continue
                self._capture_logs(name)
                _checked(self.runner, ["docker", "stop", "--time", "20", name], timeout=30)
                _remove(self.runner, name)
            except Exception as exc:
                errors.append(str(exc))
                remaining.append(container)
        self.containers = list(reversed(remaining))
        self._persist()
        self.env_file.unlink(missing_ok=True)
        if errors:
            raise RuntimeError("cleanup incomplete: " + "; ".join(errors))

    def probe_image(self) -> dict:
        """Check the pinned image on the selected pair without loading weights."""
        if self.containers:
            raise RuntimeError("stop this runtime before probing its image")
        self.topology = detect_gpus(self.config, self.runner)
        self._pins(self.config["precisions"][0])
        name = f"qwen-bench-{self.run_id}-probe"
        devices = '\"device=' + ",".join(self.topology.nvlink_pair) + '\"'
        command = ["docker", "run", "--rm", "--pull=never", "--network", "none",
                   "--name", name, "--label", f"{OWNER_LABEL}={self.run_id}",
                   "--gpus", devices, "--env", "HF_HUB_OFFLINE=1",
                   "--env", "TRANSFORMERS_OFFLINE=1", "--entrypoint", "python3",
                   self.config["docker_image"], "-c", _GPU_PROBE]
        self.containers = [{"name": name, "kind": "probe"}]
        self._persist()
        try:
            result = self.runner(command, timeout=120)
            log = (result.stdout + result.stderr)[-self.config["log_max_bytes"]:]
            (self.run_dir / "gpu-probe.log").write_text(log)
            lines = [line.split("=", 1)[1] for line in result.stdout.splitlines()
                     if line.startswith("QWEN_GPU_PROBE=")]
            if result.returncode or not lines:
                raise RuntimeError("image/GPU compatibility probe failed before model download; see gpu-probe.log")
            report = json.loads(lines[-1])
            report["nvlink_pair"] = list(self.topology.nvlink_pair)
            (self.run_dir / "gpu-probe.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        finally:
            self.stop()

    def download_models(self, candidate: Candidate | None = None) -> None:
        """Explicit, foreground download; never called by start or snapshot.

        This explicitly pulls the pinned image. Hugging Face progress writes
        directly to the invoking terminal; Ctrl-C cleans up only this downloader.
        """
        if self.containers:
            raise RuntimeError("stop this runtime's containers before downloading")
        precisions = [candidate.precision] if candidate else self.config["precisions"]
        cache = Path(self.config["cache_dir"]).expanduser().resolve()
        if any(c in str(cache) for c in (",", "\n")):
            raise ValueError("cache path must not contain commas or newlines")
        for precision in precisions:
            self._pins(precision)
        # Reject the wrong/busy host before even pulling its container image.
        self.topology = detect_gpus(self.config, self.runner)
        subprocess.run(["docker", "pull", self.config["docker_image"]], check=True)
        self.probe_image()
        cache.mkdir(parents=True, exist_ok=True)
        for precision in precisions:
            model = self._pins(precision)
            name = f"qwen-bench-{self.run_id}-download-{precision}"
            code = ("from huggingface_hub import snapshot_download; "
                    "import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                    "cache_dir='/hf/hub', ignore_patterns=['*.bin','*.pt','*.pth','*.onnx'])")
            command = ["docker", "run", "--rm", "--pull=never", "--name", name,
                       "--label", f"{OWNER_LABEL}={self.run_id}",
                       "--user", f"{os.getuid()}:{os.getgid()}",
                       "--mount", f"type=bind,src={cache},dst=/hf",
                       "--env", "HF_HOME=/hf", "--entrypoint", "python3",
                       self.config["docker_image"], "-c", code, model["repo_id"], model["revision"]]
            self.containers = [{"name": name, "kind": "download"}]
            self._persist()
            try:
                subprocess.run(command, check=True)
            finally:
                state = self._inspect(name)
                if state is not None:
                    _checked(self.runner, ["docker", "stop", "--time", "10", name], timeout=20)
                self.containers = []
                self._persist()
