import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_bench.ranking import cost, describe, shortlist
from qwen_bench.cache_check import assess


def record(layout, rate, cold=100, warm=10, precision="bf16", ident=None):
    return {"candidate": {"id": ident or layout + precision, "layout": layout, "precision": precision},
            "passed": True, "score": 1 / rate,
            "ranking": {"cold_ttft_s": cold, "warm_ttft_s": warm,
                        "short_ttft_s": .1, "max_useful_gap_s": .1}}


class RankingTests(unittest.TestCase):
    def test_double_decode_speed_wins_despite_much_faster_replica_ttft(self):
        tp2 = record("tp2", 60, cold=100, warm=30)
        replicas = record("replicas", 30, cold=1, warm=.1)
        self.assertIs(shortlist([replicas, tp2], 1, preserve_bf16=False)[0], tp2)

    def test_cache_latency_can_break_only_a_bounded_decode_speed_tie(self):
        fastest = record("tp2", 60, warm=30)
        near = record("replicas", 58, warm=1)
        slow = record("replicas", 56, warm=.01, ident="outside-band")
        self.assertIs(shortlist([slow, fastest, near], 1, preserve_bf16=False)[0], near)
        self.assertIs(shortlist([near, fastest], 1, preserve_bf16=False, speed_tolerance=0)[0], fastest)

    def test_preserving_bf16_cannot_remove_a_topology_before_long_qualification(self):
        records = [record("replicas", 90, precision="fp8"), record("replicas", 80),
                   record("tp2", 70, precision="fp8"), record("tp2", 60)]
        selected = shortlist(records, 2, preserve_layouts=True)
        self.assertEqual(len(selected), 3)
        self.assertEqual({r["candidate"]["layout"] for r in selected}, {"tp2", "replicas"})
        self.assertTrue(any(r["candidate"]["precision"] == "bf16" for r in selected))

    def test_score_uses_the_slower_user_not_aggregate_throughput(self):
        trial = {"passed": True, "concurrency": 2, "cache_mode": "distinct_prefix", "requests": [
            {"request": index, "output_tokens_per_second": rate, "ttft_s": 20, "max_useful_gap_s": .2}
            for index, rate in enumerate([20, 100, 200])]}
        self.assertEqual(cost(trial), .05)
        self.assertEqual(describe([trial])["slower_user_tokens_per_second"], 20)

    def test_missing_invalid_or_failed_measurements_cannot_win(self):
        for rate in [float("nan"), float("inf"), 0, None]:
            trial = {"passed": True, "concurrency": 2, "requests": [
                {"request": i, "output_tokens_per_second": rate} for i in range(2)]}
            self.assertEqual(cost(trial), float("inf"))
        failed = record("tp2", 100)
        failed["passed"] = False
        self.assertEqual(shortlist([failed], 2, preserve_layouts=True), [])


class CacheEvidenceTests(unittest.TestCase):
    def workload(self, backends=("worker1", "worker2")):
        return {"concurrency": 2, "requests": [{"request": i, "backend": backend,
            "expected_backend": backend, "input_tokens": 260000, "cached_input_tokens": 259000}
            for i, backend in enumerate(backends)]}

    def test_each_serving_backend_must_reuse_the_long_prefix(self):
        workload = self.workload()
        self.assertTrue(assess(workload)["passed"])
        workload["requests"][1]["cached_input_tokens"] = 0
        self.assertFalse(assess(workload)["passed"])

    def test_one_warm_user_cannot_hide_other_cold_user_on_tp2(self):
        workload = self.workload(("worker1", "worker1"))
        self.assertTrue(assess(workload)["passed"])
        workload["requests"][1]["cached_input_tokens"] = 156000
        self.assertFalse(assess(workload)["passed"])

    def test_missing_invalid_or_misrouted_usage_fails(self):
        for cached in [None, -1, True, 270000, float("nan")]:
            workload = self.workload()
            workload["requests"][0]["cached_input_tokens"] = cached
            self.assertFalse(assess(workload)["passed"])
        workload = self.workload()
        workload["requests"][0]["backend"] = "worker3"
        self.assertFalse(assess(workload)["passed"])


if __name__ == "__main__":
    unittest.main()
