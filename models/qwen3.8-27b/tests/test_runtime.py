"""Runtime safety tests: fake Docker/GPU responses, no containers or downloads."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_bench.runtime import (Candidate, DockerRuntime, OWNER_LABEL, default_config,
                                detect_gpus, generate_candidates, parse_nvlink_pairs,
                                recover, validate_config)


TOPO = """        GPU2 GPU5 GPU8 NIC0 CPU Affinity
GPU8    NV12 PHB X PIX 0-31
GPU2    X PHB NV12 PIX 0-31
GPU5    PHB X PHB PIX 0-31
"""
INVENTORY = "\n".join(f"{i}, GPU-uuid-{i}, NVIDIA A100-SXM4-80GB, 81920, 0, 0, Disabled, Disabled"
                      for i in (8, 2, 5))


class FakeRunner:
    def __init__(self):
        self.calls = []
        self.inventory = INVENTORY
        self.topology = TOPO
        self.processes = ""
        self.owners = {}
        self.running = True
        self.log = "GPU KV cache size: 700,000 tokens\n"
        self.probe_failed = False

    def __call__(self, command, timeout=30):
        self.calls.append(command)
        output, stderr, code = "", "", 0
        if command[0] == "nvidia-smi":
            output = self.topology if command[1] == "topo" else self.processes if "query-compute-apps" in command[1] else self.inventory
        elif command[:2] == ["docker", "run"]:
            name = command[command.index("--name") + 1]
            self.owners[name] = command[command.index("--label") + 1].split("=", 1)[1]
            output = "container-id"
            if name.endswith("-probe"):
                code = 1 if self.probe_failed else 0
                output = "" if self.probe_failed else 'QWEN_GPU_PROBE={"peer_access":[true,true],"transformers":"5.8.0"}'
                stderr = "incompatible CUDA driver" if self.probe_failed else ""
        elif command[:2] == ["docker", "inspect"]:
            name = command[-1]
            if name not in self.owners:
                code, stderr = 1, "Error: No such object: " + name
            elif ".State" in command[3]:
                output = json.dumps({"Running": self.running, "ExitCode": 0 if self.running else 1}) + "\n" + self.owners[name]
            else:
                output = self.owners[name]
        elif command[:2] == ["docker", "logs"]:
            output = self.log
        elif command[:2] == ["docker", "rm"]:
            self.owners.pop(command[-1], None)
        return subprocess.CompletedProcess(command, code, output, stderr)


def prepared_config(root):
    cfg = default_config()
    cfg["cache_dir"] = str(Path(root) / "cache")
    for model in cfg["models"].values():
        snapshot = Path(cfg["cache_dir"]) / "hub" / ("models--" + model["repo_id"].replace("/", "--")) / "snapshots" / model["revision"]
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "tokenizer_config.json").write_text("{}")
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"test": "model-00001.safetensors"}}))
        (snapshot / "model-00001.safetensors").write_bytes(b"fixture, not model weights")
    return cfg


class ConfigTests(unittest.TestCase):
    def test_defaults_pin_versions_and_preserve_context_headroom(self):
        cfg = default_config()
        validate_config(cfg)
        self.assertIn("@sha256:", cfg["docker_image"])
        self.assertEqual(40, len(cfg["models"]["bf16"]["revision"]))
        self.assertGreaterEqual(cfg["max_model_len"], cfg["target_input_tokens"] + cfg["output_tokens"])
        candidates = generate_candidates(cfg)
        self.assertEqual(24, len(candidates))
        self.assertEqual(len(candidates), len({c.id for c in candidates}))
        cfg["enable_mtp"] = False
        self.assertEqual({0}, {c.draft_tokens for c in generate_candidates(cfg)})

    def test_reject_unsafe_or_unsupported_configurations(self):
        for key, value in (("layouts", ["tp3"]), ("max_model_len", 262144),
                           ("kv_cache_dtype", "fp8"), ("draft_tokens", [3]),
                           ("gpu_uuids", ["GPU-a"]), ("max_num_seqs", 2),
                           ("chunks", [2049]), ("gpu_memory_utilization", 1.0)):
            cfg = default_config()
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(cfg)


class TopologyTests(unittest.TestCase):
    def test_noncontiguous_indices_reordered_rows_map_to_uuids(self):
        runner = FakeRunner()
        topo = detect_gpus(default_config(), runner)
        self.assertEqual(("GPU-uuid-2", "GPU-uuid-8"), topo.nvlink_pair)
        self.assertEqual([8, 2, 5], [g.index for g in topo.gpus])

    def test_bidirectional_nvlink_only(self):
        self.assertEqual([(2, 8)], parse_nvlink_pairs(TOPO))
        for broken in (TOPO.replace("NV12", "PHB", 1), TOPO.replace("NV12", "NV0")):
            self.assertEqual([], parse_nvlink_pairs(broken))
            runner = FakeRunner()
            runner.topology = broken
            with self.assertRaisesRegex(RuntimeError, "bidirectional"):
                detect_gpus(default_config(), runner)

    def test_busy_mig_wrong_card_and_compute_process_fail_closed(self):
        for field, value in (("inventory", INVENTORY.replace("Disabled", "Enabled", 1)),
                             ("inventory", INVENTORY.replace("81920", "40960", 1)),
                             ("inventory", INVENTORY.replace(", 0, 0,", ", 2000, 0,", 1)),
                             ("processes", "GPU-uuid-2, 12345")):
            runner = FakeRunner()
            setattr(runner, field, value)
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                detect_gpus(default_config(), runner)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = prepared_config(self.tmp.name)
        self.runner = FakeRunner()
        self.topo = detect_gpus(self.cfg, self.runner)
        self.runtime = DockerRuntime(self.cfg, Path(self.tmp.name) / "run", runner=self.runner, topology=self.topo)
        self.candidate = Candidate("bf16-tp2-c2048-mtp0", "bf16", "tp2", 2048)
        sockets = patch("qwen_bench.ports.socket.socket")
        sockets.start()
        self.addCleanup(sockets.stop)

    def test_offline_launch_preserves_vision_and_secret_never_in_argv(self):
        commands = self.runtime.launch_commands(self.candidate)
        self.assertEqual(1, len(commands))
        cmd = commands[0]
        self.assertIn('"device=GPU-uuid-2,GPU-uuid-8"', cmd)
        self.assertIn("HF_HUB_OFFLINE=1", cmd)
        self.assertIn("TRANSFORMERS_OFFLINE=1", cmd)
        self.assertIn("--pull=never", cmd)
        self.assertNotIn(self.runtime.api_key, " ".join(cmd))
        self.assertNotIn("--language-model-only", cmd)
        self.assertIn("--no-enable-log-requests", cmd)
        self.assertIn("--enable-prompt-tokens-details", cmd)
        self.assertEqual("auto", cmd[cmd.index("--kv-cache-dtype") + 1])
        self.assertEqual("bfloat16", cmd[cmd.index("--dtype") + 1])
        overrides = json.loads(cmd[cmd.index("--hf-overrides") + 1])
        self.assertEqual(2.0, overrides["text_config"]["rope_parameters"]["factor"])
        self.assertEqual({"image": 4, "video": 1}, json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]))
        self.assertEqual(0o600, self.runtime.env_file.stat().st_mode & 0o777)
        self.assertEqual([], [c for c in self.runner.calls if c[:2] == ["docker", "run"]])

    def test_replicas_map_each_uuid_once_with_fp8_and_mtp(self):
        candidate = Candidate("fp8-replicas-c8192-mtp3", "fp8", "replicas", 8192, 3)
        commands = self.runtime.launch_commands(candidate)
        self.assertEqual(3, len(commands))
        self.assertEqual(3, len({cmd[cmd.index("--gpus") + 1] for cmd in commands}))
        for cmd in commands:
            self.assertIn("--enable-prompt-tokens-details", cmd)
            self.assertEqual("1", cmd[cmd.index("--tensor-parallel-size") + 1])
            self.assertEqual("fp8", cmd[cmd.index("--quantization") + 1])
            self.assertEqual(3, json.loads(cmd[cmd.index("--speculative-config") + 1])["num_speculative_tokens"])

    def test_incomplete_snapshot_never_downloads(self):
        shard = next((Path(self.cfg["cache_dir"]) / "hub" / "models--Qwen--Qwen3.8-27B").glob("**/*.safetensors"))
        shard.unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.runtime.launch_commands(self.candidate)
        self.assertFalse(any(c[:2] in (["docker", "pull"], ["docker", "run"]) for c in self.runner.calls))

    def test_start_capacity_and_owned_cleanup(self):
        with patch.object(self.runtime, "_healthy", return_value=True):
            urls = self.runtime.start(self.candidate)
        self.assertEqual(["http://127.0.0.1:18100"], urls)
        self.assertFalse(self.runtime.env_file.exists())
        self.assertTrue(self.runtime.capacity()["passed"])
        self.assertEqual(655360, self.runtime.capacity()["required_per_engine"])
        manifest = json.loads((self.runtime.run_dir / "runtime.json").read_text())
        self.assertEqual(self.runtime.run_id, manifest["run_id"])
        self.runner.owners["unrelated-production"] = "other-owner"
        self.runtime.stop()
        self.assertEqual({"unrelated-production": "other-owner"}, self.runner.owners)
        self.assertEqual([], json.loads((self.runtime.run_dir / "runtime.json").read_text())["containers"])

    def test_early_exit_cleans_owned_containers_and_retains_logs(self):
        self.runner.running = False
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.runtime.start(self.candidate)
        self.assertEqual({}, self.runner.owners)
        self.assertTrue(list(self.runtime.run_dir.glob("*.log")))

    def test_start_retries_trailing_utilization_without_compute_process(self):
        self.runner.inventory = INVENTORY.replace(", 0, 0,", ", 0, 75,", 1)
        def settle(_seconds):
            self.runner.inventory = INVENTORY
        with patch("qwen_bench.runtime.time.sleep", side_effect=settle) as sleep, \
             patch.object(self.runtime, "_healthy", return_value=True):
            self.runtime.start(self.candidate)
        sleep.assert_called_once_with(1)
        self.runtime.stop()

    def test_port_preflight_runs_before_gpu_queries_or_docker_launch(self):
        self.runner.calls.clear()
        with patch("qwen_bench.runtime.check_ports", side_effect=RuntimeError("gateway TCP port 127.0.0.1:18080 is in use")):
            with self.assertRaisesRegex(RuntimeError, "18080"):
                self.runtime.start(self.candidate)
        self.assertEqual([], self.runner.calls)

    def test_late_docker_port_collision_reports_port_and_cleans_up(self):
        runner = self.runtime.runner
        def race(command, timeout=30):
            result = runner(command, timeout=timeout)
            if command[:2] == ["docker", "run"]:
                return subprocess.CompletedProcess(command, 125, "", "port is already allocated")
            return result
        self.runtime.runner = race
        with self.assertRaisesRegex(RuntimeError, "backend 0 on 127.0.0.1:18100: port is already allocated"):
            self.runtime.start(self.candidate)
        self.assertEqual({}, self.runner.owners)
        self.assertFalse(self.runtime.env_file.exists())

    def test_start_never_retries_an_active_compute_process(self):
        self.runner.inventory = INVENTORY.replace(", 0, 0,", ", 0, 75,", 1)
        self.runner.processes = "GPU-uuid-2, 12345"
        with patch("qwen_bench.runtime.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "active compute"):
                self.runtime.start(self.candidate)
        sleep.assert_not_called()
        self.assertFalse(any(c[:2] == ["docker", "run"] for c in self.runner.calls))

    def test_start_gives_up_after_ten_second_quiescence_budget(self):
        self.runner.inventory = INVENTORY.replace(", 0, 0,", ", 0, 75,", 1)
        with patch("qwen_bench.runtime.time.monotonic", side_effect=[0, 9, 10]), \
             patch("qwen_bench.runtime.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "still reports busy"):
                self.runtime.start(self.candidate)
        sleep.assert_called_once_with(1)
        self.assertFalse(any(c[:2] == ["docker", "run"] for c in self.runner.calls))

    def test_model_free_image_probe_checks_pair_and_records_result(self):
        report = self.runtime.probe_image()
        self.assertEqual([True, True], report["peer_access"])
        cmd = next(c for c in self.runner.calls if c[:2] == ["docker", "run"])
        self.assertIn('"device=GPU-uuid-2,GPU-uuid-8"', cmd)
        self.assertEqual("none", cmd[cmd.index("--network") + 1])
        self.assertNotIn("--mount", cmd)
        self.assertIn("torch.cuda.get_device_capability", cmd[-1])
        self.assertIn("Qwen3_5ForConditionalGeneration", cmd[-1])
        self.assertTrue((self.runtime.run_dir / "gpu-probe.json").is_file())
        self.assertEqual({}, self.runner.owners)

    def test_incompatible_image_blocks_weight_download_after_pull(self):
        self.runner.probe_failed = True
        with patch("qwen_bench.runtime.subprocess.run") as foreground:
            with self.assertRaisesRegex(RuntimeError, "before model download"):
                self.runtime.download_models(self.candidate)
        self.assertEqual(1, foreground.call_count)
        self.assertEqual(["docker", "pull"], foreground.call_args.args[0][:2])
        self.assertIn("incompatible CUDA", (self.runtime.run_dir / "gpu-probe.log").read_text())
        self.assertEqual({}, self.runner.owners)

    def test_explicit_download_uses_host_user_and_persists_before_launch(self):
        commands = []
        def foreground(command, check):
            commands.append(command)
            if command[:2] == ["docker", "run"]:
                self.assertTrue((self.runtime.run_dir / "gpu-probe.json").exists())
                manifest = json.loads((self.runtime.run_dir / "runtime.json").read_text())
                self.assertEqual("download", manifest["containers"][0]["kind"])
                self.assertEqual(command[command.index("--name") + 1], manifest["containers"][0]["name"])
            return subprocess.CompletedProcess(command, 0)
        with patch("qwen_bench.runtime.subprocess.run", side_effect=foreground):
            self.runtime.download_models(self.candidate)
        download = commands[-1]
        self.assertIn("--user", download)
        self.assertIn("HF_HOME=/hf", download)
        self.assertNotIn("readonly", download[download.index("--mount") + 1])
        self.assertEqual([], json.loads((self.runtime.run_dir / "runtime.json").read_text())["containers"])

    def test_recover_accepts_owned_downloader_and_deletes_env_copy(self):
        name = f"qwen-bench-{self.runtime.run_id}-download-bf16"
        self.runtime.containers = [{"name": name, "kind": "download"}]
        self.runtime._persist()
        self.runtime._private_write(self.runtime.env_file, "VLLM_API_KEY=fixture\n")
        self.runner.owners[name] = self.runtime.run_id
        recover(self.runtime.run_dir, self.runner)
        self.assertEqual({}, self.runner.owners)
        self.assertFalse(self.runtime.env_file.exists())

    def test_recovery_tolerates_docker_auto_remove_after_stop(self):
        name = f"qwen-bench-{self.runtime.run_id}-probe"
        self.runtime.containers = [{"name": name, "kind": "probe"}]
        self.runtime._persist()
        self.runner.owners[name] = self.runtime.run_id
        def auto_remove(command, timeout=30):
            if command[:2] == ["docker", "stop"]:
                self.runner.owners.pop(command[-1], None)
            if command[:2] == ["docker", "rm"]:
                return subprocess.CompletedProcess(command, 1, "", "Error: No such container")
            return self.runner(command, timeout=timeout)
        recover(self.runtime.run_dir, auto_remove)
        self.assertEqual([], json.loads((self.runtime.run_dir / "runtime.json").read_text())["containers"])

    def test_logs_are_bounded_redacted_and_capacity_unknown_fails(self):
        self.runner.log = self.runtime.api_key + "x" * 300000
        self.runtime.containers = [{"name": "fixture", "url": "http://127.0.0.1:18100"}]
        self.runtime._candidate = self.candidate
        self.assertFalse(self.runtime.capacity()["passed"])
        tail = (self.runtime.run_dir / "fixture.log").read_text()
        self.assertLessEqual(len(tail.encode()), self.cfg["log_max_bytes"])
        self.assertNotIn(self.runtime.api_key, tail)

    def test_recovery_refuses_changed_owner_and_keeps_manifest(self):
        with patch.object(self.runtime, "_healthy", return_value=True):
            self.runtime.start(self.candidate)
        name = self.runtime.containers[0]["name"]
        self.runner.owners[name] = "someone-else"
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            recover(self.runtime.run_dir, self.runner)
        self.assertEqual("someone-else", self.runner.owners[name])
        self.assertTrue(json.loads((self.runtime.run_dir / "runtime.json").read_text())["containers"])
        self.runner.owners[name] = self.runtime.run_id
        recover(self.runtime.run_dir, self.runner)
        self.assertEqual({}, self.runner.owners)


if __name__ == "__main__":
    unittest.main()
