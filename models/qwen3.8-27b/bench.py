#!/usr/bin/env python3
"""Run only on the target GPU server; --help and plan do not load models."""
from qwen_bench.runner import entrypoint

if __name__ == "__main__":
    raise SystemExit(entrypoint())
