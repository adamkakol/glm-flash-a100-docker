"""Shared deployment policy and config rendering; host-side standard library only."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

MODEL_CONTEXT = 1048576
TARGET_INPUT = 260000
TARGET_OUTPUT = 65536
TARGET_WINDOW = 327680
MIN_CACHE = 2 * TARGET_WINDOW
MODEL_NAME = "GLM-5.3-Flash-exl3-4.05bpw"


@dataclass(frozen=True)
class Profile:
    mode: str = "nccl"
    chunk_size: int = 2048
    draft_tokens: int = 0
    max_seq_len: int = 524288
    cache_size: int = 1048576
    max_batch_size: int = 4
    reserve_mib: int = 4096
    vision: bool = True

    def validate(self):
        if self.mode not in {"nccl", "native", "layer"}:
            raise ValueError("Unknown allocation mode")
        if self.chunk_size not in {1024, 2048, 4096}:
            raise ValueError("chunk_size must be 1024, 2048 or 4096")
        if self.draft_tokens not in {0, 1, 2}:
            raise ValueError("draft_tokens must be 0, 1 or 2")
        if type(self.vision) is not bool:
            raise ValueError("vision must be a boolean")
        validate_limits(asdict(self))
        if not 3072 <= self.reserve_mib <= 16384:
            raise ValueError("Keep 3072–16384 MiB of allocator reserve per GPU")
        return self

    @property
    def name(self):
        return (f"{self.mode}-c{self.chunk_size}-mtp{self.draft_tokens}"
                f"-ctx{self.max_seq_len}-cache{self.cache_size}-b{self.max_batch_size}-vision{int(self.vision)}")


def validate_limits(model):
    """Protect two target windows without reserving a full window for every slot."""
    seq, cache, batch = (model[k] for k in ("max_seq_len", "cache_size", "max_batch_size"))
    if any(type(v) is not int for v in (seq, cache, batch)):
        raise ValueError("Context, cache and batch limits must be integers")
    if not TARGET_WINDOW <= seq <= MODEL_CONTEXT or seq % 2048:
        raise ValueError(f"max_seq_len must be a multiple of 2048 in [{TARGET_WINDOW}, {MODEL_CONTEXT}]")
    if cache < max(MIN_CACHE, seq) or cache % 2048:
        raise ValueError(f"cache_size must be >= max({MIN_CACHE}, max_seq_len), aligned to 2048")
    if not 3 <= batch <= 8:
        raise ValueError("Use 3–8 active slots; excess requests queue when batch/cache capacity is occupied")


def render_config(template, profile):
    profile.validate()
    replacements = {
        "max_seq_len": profile.max_seq_len, "cache_size": profile.cache_size,
        "max_batch_size": profile.max_batch_size, "chunk_size": profile.chunk_size,
        "tensor_parallel": "false" if profile.mode == "layer" else "true",
        "tensor_parallel_backend": "native" if profile.mode == "layer" else profile.mode,
        "autosplit_reserve": json.dumps([profile.reserve_mib] * 3),
        "draft_mode": "mtp" if profile.draft_tokens else "disabled",
        "draft_num_tokens": max(1, profile.draft_tokens),
        "vision": str(profile.vision).lower(),
    }
    for key, value in replacements.items():
        template, count = re.subn(rf"(?m)^(  {key}:).*$", rf"\g<1> {value}", template)
        if count != 1:
            raise ValueError(f"Expected one {key} in config template, got {count}")
    return template


def read_env(root):
    path = Path(root) / ".env"
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if "=" in line and not line.startswith("#"))


def atomic_write(path, text):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def write_profile(root, profile, *, selected_by="manual", evidence=None):
    root = Path(root)
    text = render_config((root / "config.nccl.yml").read_text(), profile)
    atomic_write(root / "config.yml", text)
    record = {"profile": asdict(profile), "selected_by": selected_by,
              "config_sha256": hashlib.sha256(text.encode()).hexdigest(), "evidence": evidence}
    atomic_write(root / "deployment.json", json.dumps(record, indent=2) + "\n")
    return record
