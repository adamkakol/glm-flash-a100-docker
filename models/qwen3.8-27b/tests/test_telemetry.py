"""Controlled metric/GPU fixtures; never starts Docker or queries actual GPUs."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("qwen_telemetry", Path(__file__).resolve().parents[1] / "qwen_bench/telemetry.py")
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)
PREEMPT = "vllm:num_preemptions_total"
RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"
BACKENDS = ["http://127.0.0.1:18100", "http://127.0.0.1:18101"]


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.gpus = "GPU-a, 8192, 40, 50, 100\nGPU-b, 8192, 40, 50, 100\nGPU-unused, 100, 99, 70, 200\n"
        self.run_patch = patch.object(telemetry.subprocess, "run", side_effect=self.command)
        self.run_patch.start()

    def tearDown(self):
        self.run_patch.stop()
        self.temporary.cleanup()

    def command(self, argv, **kwargs):
        if "nvlink" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "counter unavailable")
        return subprocess.CompletedProcess(argv, 0, self.gpus, "")

    def collector(self, **kwargs):
        return telemetry.Telemetry(self.directory, BACKENDS, "test-secret", ["GPU-a", "GPU-b"], **kwargs)

    def test_parse_aggregates_labels_without_logging_key(self):
        value = telemetry.metrics('# HELP stuff\nvllm:num_requests_running{model="a"} 1\nvllm:num_requests_running{model="b"} 2\nother +Inf\n')
        self.assertEqual(value, {RUNNING: 3})

    def test_first_sample_exists_for_immediate_exit(self):
        with patch.object(telemetry, "read_metrics", return_value={PREEMPT: 3}):
            with self.collector() as result:
                pass
        summary = result.summary()
        self.assertTrue(summary["passed"], summary)
        self.assertEqual(summary["samples"], 1)
        self.assertTrue(summary["sampler_stopped"])
        self.assertFalse(summary["nvlink_traffic_verified"])
        self.assertNotIn("test-secret", (self.directory / "telemetry.jsonl").read_text())

    def test_missing_preemption_baseline_fails(self):
        responses = [{}, {}] + [{PREEMPT: 0}] * 4
        with patch.object(telemetry, "read_metrics", side_effect=responses):
            with self.collector() as result:
                pass
        self.assertFalse(result.summary()["passed"])
        self.assertFalse(result.summary()["preemption_counter_present"])

    def test_intermediate_reset_fails_even_if_counter_recovers(self):
        responses = [{PREEMPT: value} for value in (10, 10, 2, 2, 12, 12)]
        with patch.object(telemetry, "read_metrics", side_effect=responses):
            with self.collector() as result:
                pass
        self.assertFalse(result.summary()["passed"])
        self.assertFalse(result.summary()["counters_monotonic"])

    def test_missing_gpu_invalid_memory_and_low_headroom_fail(self):
        for data in ("GPU-a, 8192, 0, 50, 100\n", "GPU-a, nan, 0, 50, 100\nGPU-b, 8192, 0, 50, 100\n",
                     "GPU-a, 1024, 0, 50, 100\nGPU-b, 8192, 0, 50, 100\n"):
            with self.subTest(data=data):
                self.gpus = data
                with patch.object(telemetry, "read_metrics", return_value={PREEMPT: 0}):
                    with self.collector() as result:
                        pass
                self.assertFalse(result.summary()["passed"])

    def test_sample_gap_fails_after_an_earlier_valid_sample(self):
        with patch.object(telemetry, "read_metrics", return_value={PREEMPT: 0}):
            with self.collector() as result:
                pass
        result.started, result.sample_times, result.ended = 10, [11], 30
        self.assertFalse(result.summary()["passed"])
        self.assertEqual(result.summary()["max_sample_gap_s"], 19)

    def test_metrics_outage_is_not_erased_by_successful_final_read(self):
        responses = [{PREEMPT: 0}, {PREEMPT: 0}, OSError("fixture"), {PREEMPT: 0}, {PREEMPT: 0}, {PREEMPT: 0}]
        with patch.object(telemetry, "read_metrics", side_effect=responses):
            with self.collector() as result:
                pass
        summary = result.summary()
        self.assertTrue(summary["metrics_complete"])
        self.assertFalse(summary["passed"])
        self.assertIn("metrics: OSError", summary["errors"])

    def test_unexpected_sampler_exit_fails(self):
        with patch.object(telemetry, "read_metrics", return_value={PREEMPT: 0}):
            with self.collector(interval=.001) as result:
                with patch.object(result, "_sample", side_effect=OSError("fixture")):
                    time.sleep(.02)
        self.assertFalse(result.summary()["passed"])
        self.assertIn("sampler stopped: OSError", result.summary()["errors"])

    def test_preemption_increment_fails(self):
        responses = [{PREEMPT: value} for value in (0, 0, 0, 0, 1, 0)]
        with patch.object(telemetry, "read_metrics", side_effect=responses):
            with self.collector() as result:
                pass
        self.assertEqual(result.summary()["preemptions"], 1)
        self.assertFalse(result.summary()["passed"])

    def test_wait_idle_checks_every_backend_until_work_reclaimed(self):
        calls = []
        def read(url, key, timeout):
            calls.append(url)
            return {RUNNING: int(len(calls) == 1), WAITING: 0}
        with patch.object(telemetry, "read_metrics", side_effect=read):
            result = telemetry.wait_idle(BACKENDS, "secret", timeout=1)
        self.assertTrue(result["passed"])
        self.assertEqual(calls, BACKENDS * 2)
        self.assertEqual(set(result["backends"]), set(BACKENDS))

    def test_wait_idle_missing_invalid_or_busy_backend_fails_bounded(self):
        for value in ({RUNNING: 0}, {RUNNING: -1, WAITING: 0}, {RUNNING: 1, WAITING: 0}):
            with self.subTest(value=value), patch.object(telemetry, "read_metrics", return_value=value):
                started = time.monotonic()
                result = telemetry.wait_idle(BACKENDS, "secret", timeout=.02)
                self.assertFalse(result["passed"])
                self.assertLess(time.monotonic() - started, .2)


if __name__ == "__main__":
    unittest.main()
