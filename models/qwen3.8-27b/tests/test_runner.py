import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL))
from qwen_bench import runner
from qwen_bench.runtime import Candidate
from qwen_bench.state import atomic_json, exclusive, fingerprint, Progress


class PolicyTests(unittest.TestCase):
    def test_init_and_plan_are_offline_and_do_not_overwrite_secrets(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "config.json"
            runner.init_config(path)
            key = path.parent / "secrets/api-key"
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            before = key.read_bytes()
            with self.assertRaises(RuntimeError):
                runner.init_config(path)
            self.assertEqual(key.read_bytes(), before)
            with patch.object(runner, "detect_gpus", side_effect=AssertionError("hardware touched")), \
                    patch.object(runner.subprocess, "run", side_effect=AssertionError("process started")):
                self.assertEqual(runner.main(["--config", str(path), "plan"]), 0)

    def test_invalid_or_failed_scores_cannot_win_and_bf16_reference_retained(self):
        records = [{"passed": passed, "score": score, "candidate": {"precision": precision}}
                   for passed, score, precision in [(False, .01, "bf16"), (True, float("nan"), "fp8"),
                       (True, float("inf"), "fp8"), (True, "1", "fp8"), (True, 1, "fp8"),
                       (True, 2, "fp8"), (True, 3, "bf16")]]
        winners = runner.shortlist(records, 2)
        self.assertEqual([r["score"] for r in winners], [1, 3])

    def test_configuration_preserves_generation_headroom_and_repetition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for change in ({"output_tokens": 256}, {"repetitions": 1}, {"max_model_len": 262144},
                           {"latency_budgets": {"ttft_s": float("nan")}},
                           {"docker_image": "vllm/vllm-openai:latest"},
                           {"gateway_image": "haproxy:3.2.25-alpine"}):
                path.write_text(json.dumps({**runner.settings(), **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    runner.read_config(path)

    def test_atomic_state_private_and_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "state.json", {"value": 1})
            self.assertEqual((root / "state.json").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                atomic_json(root / "state.json", {"value": float("nan")})
            self.assertEqual(json.loads((root / "state.json").read_text()), {"value": 1})
            with exclusive(root / "lock"):
                with self.assertRaises(RuntimeError):
                    with exclusive(root / "lock"):
                        pass

    def test_constructor_failure_is_a_persisted_candidate_failure(self):
        candidate = Candidate("bf16-tp2-c2048-mtp0", "bf16", "tp2", 2048, 0)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with Progress(root / "progress", 100) as progress, \
                    patch.object(runner, "DockerRuntime", side_effect=ValueError("invalid key permissions")):
                result = runner.evaluate(runner.settings(), candidate, "screen", root / "candidate", progress)
            self.assertFalse(result["passed"])
            self.assertIn("permissions", result["error"])
            self.assertTrue((root / "candidate/result.json").exists())

    def test_recovery_does_not_require_a_config_or_key(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "REPO", Path(tmp)), \
                patch.object(runner, "recover_tree") as recover:
            self.assertEqual(runner.main(["--config", str(Path(tmp) / "absent.json"), "cleanup", tmp]), 0)
            recover.assert_called_once_with(Path(tmp))

    def test_evaluate_requires_capacity_then_repeats_real_target_and_warm_trial(self):
        candidate = Candidate("bf16-tp2-c2048-mtp0", "bf16", "tp2", 2048, 0)
        requests = [{"request": i, "output_tokens_per_second": 30, "ttft_s": 10,
                     "max_useful_gap_s": .1} for i in range(3)]
        workload = {"passed": True, "concurrency": 2, "requests": requests}
        runtime, gateway, client, telemetry = Mock(), Mock(), Mock(), Mock()
        runtime.start.return_value = ["http://127.0.0.1:18100"]
        runtime.api_key = "fixture"
        runtime.capacity.return_value = {"passed": True}
        runtime.snapshot.return_value = {}
        runtime.topology.to_dict.return_value = {"nvlink_pair": ["GPU-a", "GPU-b"]}
        gateway.start.return_value = "http://127.0.0.1:18080"
        client.smoke.side_effect = lambda: {"passed": True}
        client.long_retrieval.return_value = {"passed": True}
        client.run_workload.side_effect = lambda *a, **k: dict(workload)
        telemetry.summary.return_value = {"passed": True, "counter_deltas": {}}
        context = contextlib.nullcontext(telemetry)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, "DockerRuntime", return_value=runtime), \
                patch.object(runner, "Gateway", return_value=gateway), \
                patch.object(runner, "make_client", return_value=client), \
                patch.object(runner, "Telemetry", return_value=context), \
                patch.object(runner, "wait_idle", return_value={"passed": True}):
            result = runner.evaluate(runner.settings(), candidate, "qualify", Path(tmp), Mock())
        self.assertTrue(result["passed"], result)
        self.assertEqual(client.smoke.call_count, 3)
        self.assertEqual(client.run_workload.call_count, 3)
        calls = client.run_workload.call_args_list
        self.assertTrue(all(call.args == (260000, 32768) for call in calls))
        self.assertNotEqual(calls[0].kwargs["seed"], calls[1].kwargs["seed"])
        self.assertEqual(calls[1].kwargs["seed"], calls[2].kwargs["seed"])
        self.assertEqual([t["cache_mode"] for t in result["trials"]], ["distinct_prefix", "distinct_prefix", "reused_prefix"])
        runtime.stop.assert_called_once()
        gateway.stop.assert_called_once()

    def test_failed_workload_or_invalid_token_rate_cannot_rank(self):
        self.assertEqual(runner.cost({"passed": False}), float("inf"))
        for rate in (None, 0, float("nan")):
            result = {"passed": True, "concurrency": 1, "requests": [
                {"request": 0, "output_tokens_per_second": rate}, {"request": 1}]}
            self.assertEqual(runner.cost(result), float("inf"))


class StagedRunTests(unittest.TestCase):
    def _run(self, fail_stage=None, interrupt=False):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        config = runner.settings()
        config.update(reports_dir=str(root), precisions=["bf16"], layouts=["tp2"], chunks=[2048])
        calls = []
        def evaluation(cfg, candidate, stage, directory, progress):
            calls.append((stage, candidate.draft_tokens))
            if interrupt and stage == "refine":
                raise KeyboardInterrupt()
            passed = stage != fail_stage
            return {"candidate": candidate.__dict__, "stage": stage, "passed": passed,
                    "score": 1 + candidate.draft_tokens if passed else None}
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(runner, "detect_gpus", return_value=Mock()))
        stack.enter_context(patch.object(runner, "hardware_identity", return_value={"fixture": "stable"}))
        stack.enter_context(patch.object(runner, "evaluate", side_effect=evaluation))
        return root, config, calls

    def test_full_pipeline_recommends_only_qualified_and_reuses_completed_records(self):
        root, config, calls = self._run()
        directory = runner.tune(config)
        self.assertEqual(calls, [("screen", 0), ("refine", 0), ("refine", 1), ("refine", 3), ("qualify", 0), ("qualify", 1)])
        recommendation = json.loads((directory / "recommendation.json").read_text())
        self.assertTrue(recommendation["qualification"]["passed"])
        self.assertEqual(recommendation["candidate"]["draft_tokens"], 0)
        calls.clear()
        runner.tune(config, directory)
        self.assertEqual(calls, [])
        config["idle_timeout_s"] += 1
        with self.assertRaisesRegex(RuntimeError, "Config or source changed"):
            runner.tune(config, directory)

    def test_failed_qualification_never_exports_winner(self):
        root, config, calls = self._run(fail_stage="qualify")
        with self.assertRaisesRegex(RuntimeError, "No candidate passed full"):
            runner.tune(config)
        directory = next(root.glob("run-*"))
        self.assertFalse((directory / "recommendation.json").exists())
        self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "failed")

    def test_interruption_preserves_successful_screen_for_resume(self):
        root, config, calls = self._run(interrupt=True)
        with self.assertRaises(KeyboardInterrupt):
            runner.tune(config)
        directory = next(root.glob("run-*"))
        state = json.loads((directory / "run.json").read_text())
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(len(state["results"]), 1)
        self.assertFalse((directory / "recommendation.json").exists())

    def test_preflight_failure_is_not_left_running(self):
        root, config, calls = self._run()
        with patch.object(runner, "detect_gpus", side_effect=RuntimeError("GPU busy")):
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                runner.tune(config)
        record = json.loads(next(root.glob("run-*/run.json")).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
