"""Require attributable cache reuse for each individual follow-up request."""
from __future__ import annotations


def assess(workload, minimum_ratio=.8):
    errors, measurements = [], []
    requests = [r for r in workload.get("requests", []) if r["request"] < workload["concurrency"]]
    if len(requests) != workload["concurrency"]:
        errors.append("missing long-session requests")
    for request in requests:
        tokens, cached = request.get("input_tokens"), request.get("cached_input_tokens")
        valid = type(tokens) is int and tokens > 0 and type(cached) is int and 0 <= cached <= tokens
        ratio = cached / tokens if valid else None
        backend = request.get("backend")
        if not backend or backend != request.get("expected_backend"):
            errors.append("follow-up did not preserve its session backend")
        measurements.append({"request": request["request"], "backend": backend,
                             "cached_tokens": cached, "input_tokens": tokens, "ratio": ratio})
        if not valid:
            errors.append("missing/invalid per-request cached-token usage; check --enable-prompt-tokens-details")
        elif ratio < minimum_ratio:
            errors.append("per-request prefix-cache reuse below required ratio")
    return {"passed": bool(requests) and not errors, "minimum_ratio": minimum_ratio,
            "requests": measurements, "errors": errors,
            "scope": "Server-reported cached tokens for each growing follow-up; aggregate counters cannot substitute."}
