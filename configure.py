#!/usr/bin/env python3
"""Host configuration: standard library only; run as your normal Docker user."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from scripts.deployment import Profile, read_env, write_profile

ROOT = Path(__file__).resolve().parent


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30)


def parse_gpus(text):
    result = {}
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        idx, uuid, name, memory, driver, mig = [v.strip() for v in row]
        result[int(idx)] = dict(index=int(idx), uuid=uuid, name=name,
                                memory_mib=float(memory), driver=driver, mig=mig)
    return result


def nvlink_pairs(text):
    lines = text.splitlines()
    header = next((line for line in lines if re.match(r"\s*GPU\d+\s+GPU\d+", line)), "")
    labels = re.findall(r"\bGPU\d+\b", header)
    pairs = set()
    for line in lines:
        cells = line.split()
        if not cells or cells[0] not in labels:
            continue
        src = int(cells[0][3:])
        for label, link in zip(labels, cells[1:]):
            if re.fullmatch(r"NV\d+", link):
                pairs.add(tuple(sorted((src, int(label[3:])))))
    return sorted(pairs)


def select_order(gpus, pairs, selected):
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("Select exactly three distinct GPU indices with --gpus, for example 0,1,2.")
    for idx in selected:
        if idx not in gpus:
            raise ValueError(f"GPU {idx} was not found.")
        gpu = gpus[idx]
        if "A100" not in gpu["name"] or gpu["memory_mib"] < 70000:
            raise ValueError(f"GPU {idx}: this 4.05bpw deployment requires an A100 80GB; found {gpu['name']} with {gpu['memory_mib']} MiB.")
        if gpu["mig"].lower() == "enabled":
            raise ValueError(f"GPU {idx} has MIG enabled; this deployment requires whole GPUs.")
        version = tuple(int(x) for x in gpu["driver"].split("."))
        if version < (570, 124, 6):
            raise ValueError("This bundle requires driver >=570.124.06 for its CUDA 12.8.1/Triton setup.")
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu["uuid"]):
            raise ValueError("Unexpected GPU UUID format.")
    matches = [pair for pair in pairs if set(pair).issubset(selected)]
    if not matches:
        raise ValueError("No NVLink pair was found among the selected GPUs. Check nvidia-smi topo -m and the bridge.")
    pair = matches[0]
    return [*pair, next(i for i in selected if i not in pair)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", help="Three host nvidia-smi GPU indices; auto-select if exactly three GPUs exist.")
    parser.add_argument("--mode", choices=["nccl", "native", "layer"])
    parser.add_argument("--port", type=int)
    for option in ("max-seq-len", "cache-size", "max-batch-size", "chunk-size", "draft-tokens"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--force", action="store_true", help="Replace generated .env and config.yml, retaining existing keys/data.")
    args = parser.parse_args()
    previous_env = read_env(ROOT)
    port = args.port if args.port is not None else int(previous_env.get("API_PORT", "5000"))
    if not 1024 <= port <= 65535:
        parser.error("Choose a port from 1024 to 65535.")
    if not args.force and any((ROOT / p).exists() for p in [".env", "config.yml"]):
        parser.error("Existing configuration found; use --force to replace it (keys and model data are retained).")
    gpus = parse_gpus(command("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,mig.mode.current", "--format=csv,noheader,nounits"))
    topo = command("nvidia-smi", "topo", "-m")
    if args.gpus:
        selected = [int(x) for x in args.gpus.split(",")]
    elif all(f"GPU{i}_UUID" in previous_env for i in range(3)):
        by_uuid = {gpu["uuid"]: idx for idx, gpu in gpus.items()}
        try:
            selected = [by_uuid[previous_env[f"GPU{i}_UUID"]] for i in range(3)]
        except KeyError:
            raise ValueError("A previously selected GPU is missing; select GPUs explicitly with --gpus")
    else:
        selected = list(gpus)
    order = select_order(gpus, nvlink_pairs(topo), selected)
    print(topo)
    for logical, physical in enumerate(order):
        print(f"CUDA {logical} -> host GPU {physical}: {gpus[physical]['uuid']} ({gpus[physical]['name']})")
    for directory in ["secrets", "data/models", "data/cache", "data/logs", "reports"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    os.chmod(ROOT / "secrets", 0o700)
    keys = ROOT / "secrets/api_tokens.yml"
    if not keys.exists():
        # JSON is valid YAML; this also makes local API clients dependency-free.
        with keys.open("x") as handle:
            os.chmod(keys, 0o600)
            json.dump({"api_key": [secrets.token_urlsafe(32), secrets.token_urlsafe(32)],
                       "admin_key": secrets.token_urlsafe(48)}, handle, indent=2)
            handle.write("\n")
    previous = ROOT / "deployment.json"
    fields = json.loads(previous.read_text())["profile"] if previous.exists() else {}
    for name in ("mode", "max_seq_len", "cache_size", "max_batch_size", "chunk_size", "draft_tokens"):
        if getattr(args, name) is not None:
            fields[name] = getattr(args, name)
    profile = Profile(**fields).validate()
    write_profile(ROOT, profile)
    env = [f"GPU{i}_UUID={gpus[idx]['uuid']}" for i, idx in enumerate(order)]
    env += [f"LOCAL_UID={os.getuid()}", f"LOCAL_GID={os.getgid()}", f"API_PORT={port}"]
    (ROOT / ".env").write_text("\n".join(env) + "\n")
    (ROOT / "reports/host-topology.txt").write_text(topo)
    (ROOT / "reports/gpu-selection.json").write_text(json.dumps([gpus[i] for i in order], indent=2))
    print(f"Configured {profile.name}, localhost port={port}. Keys: secrets/api_tokens.yml (not printed).")
    print("Next: docker compose build")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
