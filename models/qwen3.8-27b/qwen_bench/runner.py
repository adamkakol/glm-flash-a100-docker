"""Staged qualification and explicit deployment of a measured candidate."""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.request

from .benchmark import BenchmarkClient
from .cache_check import assess as assess_cache
from .gateway import Gateway, recover as recover_gateway
from .ranking import POLICY, cost, describe, shortlist
from .ports import check_ports, validate_port_layout
from .runtime import Candidate, DockerRuntime, default_config, detect_gpus, generate_candidates, recover as recover_runtime, validate_config
from .state import Progress, atomic_json, exclusive, fingerprint
from .telemetry import Telemetry, wait_idle

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
GATEWAY_IMAGE = "haproxy:3.2.25-alpine@sha256:5d97434a423c2533cfeb42874d45cf6d69a840b60334582dfbfd5008e94c80be"


def settings() -> dict:
    value = default_config()
    value.update({"gateway_image": GATEWAY_IMAGE, "gateway_port": 18080,
        "gateway_backend_maxconn": 4, "gateway_backend_maxqueue": 8,
        "gateway_maxconn": 64, "gateway_timeout_queue_s": 30,
        "gateway_timeout_client_s": 900, "gateway_timeout_server_s": 900,
        "screen_input_tokens": 8192, "screen_output_tokens": 512,
        "refine_input_tokens": 65536, "refine_output_tokens": 1024,
        "finalists": 2, "repetitions": 2, "warm_prefix_trial": True,
        "decode_speed_tolerance": .05, "minimum_prefix_cache_hit_ratio": .8,
        "gateway_session_table_size": 10000, "gateway_session_ttl_s": 86400,
        "smoke_repetitions": 3, "request_timeout_s": 7200,
        "idle_timeout_s": 300, "latency_budgets": {"ttft_s": 300, "max_useful_gap_s": 120},
        "minimum_free_gib": 3, "progress_interval_s": 15,
        "reasoning_effort": "xhigh", "enable_mtp": True,
        "max_num_seqs": 4, "reports_dir": "reports", "api_key_file": "secrets/api-key",
        "cache_dir": "data/huggingface"})
    return value


def read_config(path: Path) -> dict:
    config = settings()
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError("configuration must be a JSON object")
    config.update(supplied)
    for key in ("docker_image", "gateway_image"):
        if not isinstance(config[key], str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9./_:@-]*@sha256:[a-f0-9]{64}", config[key]):
            raise ValueError(key + " must have an immutable sha256 digest")
    for key in ("cache_dir", "api_key_file", "reports_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    validate_config(config)
    validate_port_layout(config)
    for key in ("screen_input_tokens", "screen_output_tokens", "refine_input_tokens", "refine_output_tokens",
                "finalists", "repetitions", "smoke_repetitions", "request_timeout_s", "idle_timeout_s", "progress_interval_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if type(config["decode_speed_tolerance"]) not in (int, float) or not 0 <= config["decode_speed_tolerance"] <= .1:
        raise ValueError("decode_speed_tolerance must be between 0 and 0.1")
    if type(config["minimum_prefix_cache_hit_ratio"]) not in (int, float) or not .5 <= config["minimum_prefix_cache_hit_ratio"] <= 1:
        raise ValueError("minimum_prefix_cache_hit_ratio must be between 0.5 and 1")
    if config["warm_prefix_trial"] is not True:
        raise ValueError("warm_prefix_trial must remain enabled for session-cache qualification")
    if config["repetitions"] < 2 or config["smoke_repetitions"] < 2:
        raise ValueError("qualification requires at least two repeated workload and smoke trials")
    if config["target_input_tokens"] < 260000 or config["output_tokens"] < 32768:
        raise ValueError("qualification must preserve 260000 input plus at least 32768 output tokens per user")
    if config["target_input_tokens"] + config["output_tokens"] > config["max_model_len"]:
        raise ValueError("context must include input and generation")
    for prefix in ("screen", "refine"):
        if config[prefix + "_input_tokens"] + config[prefix + "_output_tokens"] > config["max_model_len"]:
            raise ValueError(prefix + " workload exceeds context")
    if config["reasoning_effort"] not in ("xhigh", "medium", "low"):
        raise ValueError("invalid reasoning_effort")
    if not isinstance(config["latency_budgets"], dict) or any(
            not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in config["latency_budgets"].values()):
        raise ValueError("latency budgets must be finite positive numbers")
    if not {"ttft_s", "max_useful_gap_s"} <= config["latency_budgets"].keys():
        raise ValueError("TTFT and useful-gap budgets are required")
    if not isinstance(config["minimum_free_gib"], (float, int)) or not 1 <= config["minimum_free_gib"] <= 20:
        raise ValueError("minimum_free_gib must be between 1 and 20")
    if config["gateway_backend_maxconn"] > config["max_num_seqs"]:
        raise ValueError("gateway concurrency must not exceed engine active sequence limit")
    return config


def init_config(path: Path):
    if path.exists():
        raise RuntimeError("Refusing to overwrite existing config: " + str(path))
    config = settings()
    key_path = path.parent / config["api_key_file"]
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if not key_path.exists():
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(48) + "\n")
    atomic_json(path, config)
    print("Created", path, "and API key file", key_path, "(key not printed).")


def fixture_url() -> str:
    return "data:video/mp4;base64," + base64.b64encode((ROOT / "fixtures/red-blue.mp4").read_bytes()).decode()


def make_client(config, url, key, progress=None):
    return BenchmarkClient(url, api_key=key, timeout=config["request_timeout_s"],
        idle_timeout=config["idle_timeout_s"], progress=progress,
        latency_budgets=config["latency_budgets"], video_fixture_url=fixture_url())


def recover_tree(directory: Path):
    for manifest in directory.rglob("gateway.json"):
        recover_gateway(manifest.parent)
    for manifest in directory.rglob("runtime.json"):
        recover_runtime(manifest.parent)


def compare_report(directory: Path):
    """Inspect saved results without loading GPUs, changing a winner, or writing files."""
    state = json.loads((directory / "run.json").read_text())
    comparisons = []
    for name, result in state.get("results", {}).items():
        try:
            metrics = describe(result.get("trials", []))
        except (ValueError, KeyError, TypeError) as exc:
            metrics = {"unavailable": str(exc)}
        comparisons.append({"stage_candidate": name, "passed_original_checks": result.get("passed", False),
                            "candidate": result.get("candidate"), "measured_metrics": metrics,
                            "session_cache_qualified": bool(result.get("ranking", {}).get("policy") == POLICY
                                and result.get("passed") and result.get("stage") == "qualify")})
    print(json.dumps({"read_only": True, "original_winner": state.get("winner"),
        "note": "Per-user measurements, not the legacy blended score. This does not qualify or deploy an old winner under the new routing policy.",
        "results": comparisons}, indent=2, allow_nan=False))


def hardware_identity(topology) -> dict:
    """Persist stable identity, not idle utilization or JSON tuple/list artifacts."""
    raw = topology.to_dict()
    fields = ("index", "uuid", "name", "memory_mib", "mig_current", "mig_pending")
    value = {"gpus": [{k: g[k] for k in fields} for g in raw["gpus"]],
             "nvlink_pair": list(raw["nvlink_pair"]), "topology_text": raw["topology_text"]}
    output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15).stdout
    selected = {g["uuid"] for g in value["gpus"]}
    value["drivers"] = {line.split(",")[0].strip(): line.split(",")[1].strip()
                        for line in output.splitlines() if "," in line and line.split(",")[0].strip() in selected}
    if len(value["drivers"]) != len(selected):
        raise RuntimeError("Could not record every selected GPU driver")
    return value


def evaluate(config: dict, candidate: Candidate, stage: str, directory: Path, progress: Progress) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    runtime = gateway = client = telemetry = None
    result = {"candidate": asdict(candidate), "stage": stage, "passed": False,
              "score": None, "started_at": time.time(), "trials": [], "smoke": []}
    progress.update(candidate=candidate.id, activity={"event": "starting_container"})
    progress.log(stage + ": " + candidate.id)
    try:
        runtime = DockerRuntime(config, directory)
        gateway = Gateway(config, directory)
        backends = runtime.start(candidate)
        result["runtime"] = runtime.snapshot()
        result["capacity"] = runtime.capacity()
        if not result["capacity"].get("passed"):
            raise RuntimeError("Observed KV capacity does not cover both target windows; see capacity report")
        url = gateway.start(backends)
        client = make_client(config, url, runtime.api_key, progress.event)
        for _ in range(config["smoke_repetitions"] if stage == "qualify" else 1):
            check = client.smoke()
            result["smoke"].append(check)
            if not check["passed"]:
                raise RuntimeError("Protocol, modality, or known-answer gate failed; see smoke report")
            check["idle_after_cancellation"] = wait_idle(backends, runtime.api_key, timeout=30)
            if not check["idle_after_cancellation"]["passed"]:
                raise RuntimeError("Cancelled/completed requests did not release every backend within 30 seconds")
        # Only selected GPUs are assessed; an idle third GPU is not a TP2 failure.
        topology = runtime.topology.to_dict()
        uuids = list(topology["nvlink_pair"] if candidate.layout == "tp2" else [g["uuid"] for g in topology["gpus"]])
        with Telemetry(directory, backends, runtime.api_key, uuids) as telemetry:
            if stage == "qualify":
                progress.update(activity={"event": "long_context_correctness"})
                result["long_retrieval"] = client.long_retrieval(config["target_input_tokens"], seed=98761,
                                                                reasoning_effort=config["reasoning_effort"])
                if not result["long_retrieval"]["passed"]:
                    raise RuntimeError("Long-context retrieval correctness failed")
                input_tokens, output_tokens = config["target_input_tokens"], config["output_tokens"]
                repetitions = config["repetitions"]
            else:
                input_tokens, output_tokens = config[stage + "_input_tokens"], config[stage + "_output_tokens"]
                repetitions = 1
            for repetition in range(repetitions):
                # Identical seeds between candidates, distinct prefixes between users/trials.
                seed = 31000 + repetition * 1000003
                progress.update(activity={"event": "cold_trial", "repetition": repetition + 1})
                trial = client.run_workload(input_tokens, output_tokens, seed=seed,
                                            reasoning_effort=config["reasoning_effort"])
                trial["cache_mode"] = "distinct_prefix"
                result["trials"].append(trial)
                atomic_json(directory / "result.json", result)
                if not trial["passed"]:
                    raise RuntimeError("Concurrent workload failed; see per-request metrics")
                if stage == "qualify":
                    progress.update(activity={"event": "session_followup_trial", "repetition": repetition + 1})
                    warm = client.run_followup_workload(input_tokens, config["refine_output_tokens"], seed=seed,
                        reasoning_effort=config["reasoning_effort"], max_model_len=config["max_model_len"])
                    warm["cache_mode"] = "growing_followup"
                    result["trials"].append(warm)
                    if not warm["passed"]:
                        raise RuntimeError("Growing-conversation workload or session affinity failed")
                    warm["prefix_cache"] = assess_cache(warm, config["minimum_prefix_cache_hit_ratio"])
                    atomic_json(directory / "result.json", result)
                    if not warm["prefix_cache"]["passed"]:
                        raise RuntimeError("Growing conversations did not reuse sufficient cached prefix tokens")
        result["telemetry"] = telemetry.summary(config["minimum_free_gib"])
        if not result["telemetry"]["passed"]:
            raise RuntimeError("VRAM headroom, metrics, or no-preemption gate failed")
        if candidate.draft_tokens:
            counters = result["telemetry"]["counter_deltas"]
            drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0)
            accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0)
            result["speculation"] = {"drafted": drafted, "accepted": accepted,
                                      "acceptance": accepted / drafted if drafted else None}
            if drafted <= 0:
                raise RuntimeError("Speculation requested but no draft tokens observed")
        result["ranking"] = describe(result["trials"])
        score = result["ranking"]["seconds_per_output_token"]
        if not math.isfinite(score):
            raise RuntimeError("Missing or invalid ranking metrics")
        result.update(passed=True, score=score)
    except Exception as exc:
        result["error"] = str(exc)
        progress.log(candidate.id + " failed: " + str(exc))
    finally:
        if client:
            client.close()
        if telemetry is not None and "telemetry" not in result:
            result["telemetry"] = telemetry.summary(config["minimum_free_gib"])
        cleanup_errors = []
        for component in (gateway, runtime):
            if component:
                try:
                    component.stop()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
        if cleanup_errors:
            result.update(passed=False, score=None, cleanup_errors=cleanup_errors)
        result["finished_at"] = time.time()
        atomic_json(directory / "result.json", result)
        if cleanup_errors:
            raise RuntimeError("Cleanup incomplete; run cleanup on " + str(directory))
    return result


def tune(config: dict, resume: Path | None = None, retry_failed=False) -> Path:
    signature = fingerprint(config, ROOT)
    directory = resume or Path(config["reports_dir"]) / (time.strftime("run-%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3))
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    state_path = directory / "run.json"
    if resume:
        state = json.loads(state_path.read_text())
        if state["fingerprint"] != signature:
            raise RuntimeError("Config or source changed: create a new run; mixed-version resume is unsafe")
        recover_tree(directory)
        if retry_failed:
            state["results"] = {k: v for k, v in state["results"].items() if v.get("passed")}
    else:
        state = {"schema": 1, "fingerprint": signature, "config": config, "results": {}, "status": "running"}
        atomic_json(state_path, state)
    try:
        check_ports(config, backend_count=3 if "replicas" in config["layouts"] else 1)
        current = hardware_identity(detect_gpus(config))
        if state.get("hardware") and state["hardware"] != current:
            raise RuntimeError("Hardware/driver/topology changed; create a new run")
    except BaseException as exc:
        state.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        atomic_json(state_path, state)
        raise
    state["hardware"] = current
    atomic_json(state_path, state)
    candidates = generate_candidates(config)
    with Progress(directory, config["progress_interval_s"]) as progress:
        def stage(name, chosen):
            results = []
            progress.update(stage=name, completed=0, total=len(chosen))
            for index, candidate in enumerate(chosen):
                key = name + "/" + candidate.id
                if key in state["results"]:
                    result = state["results"][key]
                    progress.log("Resume: reusing completed " + key)
                else:
                    result = evaluate(config, candidate, name, directory / name / candidate.id, progress)
                    state["results"][key] = result
                    atomic_json(state_path, state)
                results.append(result)
                progress.update(completed=index + 1)
            return results
        try:
            screen = stage("screen", [c for c in candidates if c.draft_tokens == 0])
            selected = shortlist(screen, config["finalists"], preserve_layouts=True,
                                 speed_tolerance=config["decode_speed_tolerance"])
            if not selected:
                raise RuntimeError("No candidate passed screening; no deployment will be exported")
            identities = {(r["candidate"]["precision"], r["candidate"]["layout"], r["candidate"]["chunk_size"]) for r in selected}
            refined = stage("refine", [c for c in candidates if (c.precision, c.layout, c.chunk_size) in identities])
            finalists = [Candidate(**r["candidate"]) for r in shortlist(refined, config["finalists"], preserve_layouts=True,
                            speed_tolerance=config["decode_speed_tolerance"])]
            if not finalists:
                raise RuntimeError("No candidate passed refinement")
            qualified = stage("qualify", finalists)
            ranked = shortlist(qualified, len(qualified), preserve_bf16=False,
                               speed_tolerance=config["decode_speed_tolerance"])
            if not ranked:
                raise RuntimeError("No candidate passed full qualification; no production recommendation")
            state.update(status="completed", winner=ranked[0]["candidate"])
            recommendation = {"schema": 1, "fingerprint": signature, "config": config,
                "hardware": state["hardware"], "candidate": state["winner"],
                "qualification": ranked[0], "report_directory": str(directory.resolve()),
                "selection_policy": {"name": POLICY, "decode_speed_tolerance": config["decode_speed_tolerance"],
                    "primary": "slower-user generation rate across cold trials",
                    "near_speed_ties": "warm TTFT, cold TTFT, useful gap, third-request TTFT"},
                "qualified_comparison": [{"candidate": r["candidate"], "ranking": r.get("ranking")} for r in ranked],
                "scope": "Synthetic protocol, multimodal, retrieval and load qualification; not a general intelligence benchmark."}
            atomic_json(directory / "recommendation.json", recommendation)
            progress.update(status="completed", stage="completed", candidate=state["winner"]["id"])
            progress.log("Qualified recommendation: " + str(directory / "recommendation.json"))
        except BaseException as exc:
            state["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            state["error"] = type(exc).__name__ + ": " + str(exc)
            progress.update(status=state["status"])
            raise
        finally:
            atomic_json(state_path, state)
    return directory


def serve(report: Path, config: dict, port: int):
    recommendation = json.loads(report.read_text())
    if recommendation["fingerprint"] != fingerprint(config, ROOT):
        raise RuntimeError("Config/source differs from qualified recommendation; rerun qualification")
    if not recommendation.get("qualification", {}).get("passed"):
        raise RuntimeError("Recommendation is not qualified")
    check_ports(config, backend_count=1 if recommendation["candidate"]["layout"] == "tp2" else 3,
                gateway_port=port)
    topology = detect_gpus(config)
    if hardware_identity(topology) != recommendation["hardware"]:
        raise RuntimeError("Hardware/driver differs from the qualified run")
    directory = ROOT / "state" / ("serve-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3))
    directory.mkdir(parents=True, mode=0o700)
    serving = {**config, "gateway_port": port, "runtime_restart": "unless-stopped"}
    runtime, gateway, client = DockerRuntime(serving, directory), Gateway(serving, directory), None
    try:
        backends = runtime.start(Candidate(**recommendation["candidate"]))
        url = gateway.start(backends)
        client = make_client(serving, url, runtime.api_key)
        check = client.smoke()
        check["idle_after_cancellation"] = wait_idle(backends, runtime.api_key, timeout=30)
        check["passed"] = check["passed"] and check["idle_after_cancellation"]["passed"]
        atomic_json(directory / "smoke.json", check)
        if not check["passed"]:
            raise RuntimeError("Post-deployment smoke failed; see " + str(directory / "smoke.json"))
        atomic_json(ROOT / "state/active.json", {"directory": str(directory), "url": url,
                    "recommendation": str(report.resolve()), "candidate": recommendation["candidate"]})
        print("Serving", url + "/v1", "model=qwen3.8-27b; key file:", config["api_key_file"])
        print("State and logs:", directory)
    except BaseException:
        try:
            gateway.stop()
        finally:
            runtime.stop()
        raise
    finally:
        if client:
            client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "plan", "preflight", "download", "status", "stop"):
        commands.add_parser(name)
    run = commands.add_parser("run")
    run.add_argument("--resume", type=Path)
    run.add_argument("--retry-failed", action="store_true", help="Retry failed records in a resumed run; keep successful records")
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("directory", type=Path)
    comparison = commands.add_parser("compare", help="Read saved per-user rates without starting containers or changing a recommendation")
    comparison.add_argument("directory", type=Path)
    deployment = commands.add_parser("serve")
    deployment.add_argument("recommendation", type=Path)
    deployment.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "init":
        init_config(args.config.resolve())
        return 0
    if args.command == "compare":
        compare_report(args.directory.resolve())
        return 0
    if args.command == "status":
        active = ROOT / "state/active.json"
        if active.exists():
            record = json.loads(active.read_text())
            try:
                with urllib.request.urlopen(record["url"] + "/health", timeout=3) as response:
                    record["gateway_healthy_now"] = response.status == 200
            except OSError:
                record["gateway_healthy_now"] = False
            print(json.dumps(record, indent=2))
        config = read_config(args.config.resolve())
        runs = sorted(Path(config["reports_dir"]).glob("run-*/progress.json"))
        print(runs[-1].read_text() if runs else "No benchmark reports yet.")
        return 0
    if args.command in ("cleanup", "stop"):
        # Recovery must work even if config.json or the API key was lost/edited.
        with exclusive(REPO / ".gpu-benchmark.lock"):
            if args.command == "cleanup":
                recover_tree(args.directory.resolve())
            else:
                active = ROOT / "state/active.json"
                if active.exists():
                    state = json.loads(active.read_text())
                    recover_tree(Path(state["directory"]))
                    active.unlink()
                    print("Stopped managed Qwen deployment.")
        return 0
    config = read_config(args.config.resolve())
    if args.command == "plan":
        print(json.dumps({"downloads": False, "starts_containers": False, "config": config,
              "candidates": [asdict(c) for c in generate_candidates(config)],
              "stages": ["screen: speculation off", "refine: top layouts with/without MTP",
                         "qualify: repeated two long sessions, mixed third request, modalities and tools"],
              "qualification_output_tokens_per_user": config["output_tokens"]}, indent=2))
        return 0
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    with exclusive(REPO / ".gpu-benchmark.lock"):
        if args.command == "preflight":
            ports = check_ports(config, backend_count=3 if "replicas" in config["layouts"] else 1)
            topology = detect_gpus(config)
            print(json.dumps({**topology.to_dict(), "ports": ports}, indent=2))
            subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True, timeout=30)
        elif args.command == "download":
            # Target hardware guard comes before any image/model transfer.
            topology = detect_gpus(config)
            directory = ROOT / "state/download"
            directory.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "pull", "--platform", "linux/amd64", config["gateway_image"]], check=True)
            DockerRuntime(config, directory, topology=topology).download_models()
        elif args.command == "run":
            if args.retry_failed and not args.resume:
                raise ValueError("--retry-failed requires --resume")
            print("Report:", tune(config, args.resume.resolve() if args.resume else None, args.retry_failed))
        elif args.command == "serve":
            if not 1024 <= args.port <= 65535:
                raise ValueError("Serve port must be 1024..65535")
            if (ROOT / "state/active.json").exists():
                raise RuntimeError("A deployment is recorded; inspect status and stop it explicitly before replacing it")
            serve(args.recommendation.resolve(), config, args.port)
    return 0


def entrypoint():
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted. Managed containers were stopped; completed reports can be resumed.", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1
