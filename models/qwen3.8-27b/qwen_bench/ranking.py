"""Two-user interactive ranking, with explicit bounds on decode tradeoffs."""
from __future__ import annotations

import math
import statistics


POLICY = "two-user-decode-v2"


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def cost(workload: dict) -> float:
    """Seconds per output token for the slower long-session user, never aggregate TPS."""
    if not workload.get("passed"):
        return math.inf
    long = [r for r in workload.get("requests", []) if r["request"] < workload["concurrency"]]
    if len(long) != workload["concurrency"]:
        return math.inf
    rates = [r.get("output_tokens_per_second") for r in long]
    return 1 / min(rates) if rates and all(positive(v) for v in rates) else math.inf


def describe(trials: list[dict]) -> dict:
    cold = [t for t in trials if t.get("cache_mode") == "distinct_prefix"]
    warm = [t for t in trials if t.get("cache_mode") == "growing_followup"]
    costs = [cost(t) for t in cold]
    if not costs or not all(positive(v) for v in costs):
        raise ValueError("Missing or invalid two-user decode measurements")

    def median_worst(workloads, metric, short=False):
        values = []
        for trial in workloads:
            group = [r for r in trial["requests"]
                     if (r["request"] >= trial["concurrency"]) == short]
            if group:
                measurements = [r.get(metric) for r in group]
                if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in measurements):
                    raise ValueError("Missing or invalid " + metric)
                values.append(max(measurements))
        return statistics.median(values) if values else None

    return {"policy": POLICY, "seconds_per_output_token": statistics.median(costs),
            "slower_user_tokens_per_second": 1 / statistics.median(costs),
            "cold_ttft_s": median_worst(cold, "ttft_s"),
            "warm_ttft_s": median_worst(warm, "ttft_s"),
            "short_ttft_s": median_worst(cold, "ttft_s", short=True),
            "max_useful_gap_s": median_worst(cold + warm, "max_useful_gap_s"),
            "cold_trials": len(cold), "warm_trials": len(warm)}


def shortlist(results: list[dict], count: int, preserve_bf16=True,
              preserve_layouts=False, speed_tolerance=.05) -> list[dict]:
    """Latency breaks near-speed ties; preserve viable layouts before qualification.

    The count is a minimum when preserving layout/BF16 coverage requires more
    candidates. Final selection disables both preservation rules.
    """
    if not 0 <= speed_tolerance <= .10:
        raise ValueError("decode speed tolerance must be between zero and 10 percent")
    remaining = [r for r in results if r.get("passed") and positive(r.get("score"))]
    ranked = []
    def latency_key(record):
        metrics = record.get("ranking", {})
        def metric(name):
            value = metrics.get(name)
            return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else math.inf
        return (metric("warm_ttft_s"), metric("cold_ttft_s"), metric("max_useful_gap_s"),
                metric("short_ttft_s"), record["score"], record["candidate"].get("id", ""))
    while remaining:
        fastest_cost = min(r["score"] for r in remaining)
        band = [r for r in remaining if r["score"] <= fastest_cost / (1 - speed_tolerance)]
        ranked.extend(sorted(band, key=latency_key))
        remaining = [r for r in remaining if r not in band]
    if not preserve_layouts:
        selected = ranked[:count]
        baseline = next((r for r in ranked if r["candidate"]["precision"] == "bf16"), None)
        if preserve_bf16 and count > 1 and baseline and baseline not in selected:
            selected[-1:] = [baseline]
        return selected
    selected, layouts = [], set()
    for record in ranked:
        layout = record["candidate"]["layout"]
        if layout not in layouts:
            selected.append(record)
            layouts.add(layout)
    if preserve_bf16 and not any(r["candidate"]["precision"] == "bf16" for r in selected):
        baseline = next((r for r in ranked if r["candidate"]["precision"] == "bf16"), None)
        if baseline:
            selected.append(baseline)
    for record in ranked:
        if len(selected) >= count:
            break
        if record not in selected:
            selected.append(record)
    return selected
