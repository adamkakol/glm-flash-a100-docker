import argparse
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import autotune
import configure
from scripts.deployment import Profile, validate_limits, write_profile
from scripts.api_check import summarize
from scripts.gpu_topology import nvlink_pairs, require_nvlink_pair


def record(name, ttft=100, speed=20, extra=5, pause=.5, status='passed'):
    return {'name': name, 'profile': asdict(Profile(mode=name)), 'status': status, 'trials': [
        {'long': {'metrics': {'worst_ttft_s': ttft, 'worst_decode_tokens_per_s': speed}},
         'mixed': {'metrics': {'extra_user_worst_ttft_s': extra, 'worst_max_stream_gap_s': pause}}}]}


class PolicyTests(unittest.TestCase):
    def test_shared_pool_does_not_require_full_window_per_slot(self):
        profile = replace(Profile(), max_seq_len=1048576, cache_size=1572864, max_batch_size=4)
        profile.validate()
        self.assertLess(profile.cache_size, profile.max_batch_size * profile.max_seq_len)

    def test_guards_protect_target_and_native_context(self):
        for changes in [{'cache_size': 524288}, {'max_batch_size': 2}, {'max_seq_len': 262144},
                        {'max_seq_len': 2097152}, {'cache_size': 1048577}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(Profile(), **changes).validate()

    def test_render_all_modes_and_drafting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'config.nccl.yml').write_text((ROOT / 'config.nccl.yml').read_text())
            for mode in ['nccl', 'native', 'layer']:
                for draft in [0, 1, 2]:
                    profile = replace(Profile(), mode=mode, draft_tokens=draft)
                    record = write_profile(root, profile)
                    self.assertEqual(record['profile'], asdict(profile))
                    text = (root / 'config.yml').read_text()
                    self.assertIn('tensor_parallel: ' + ('false' if mode == 'layer' else 'true'), text)
                    self.assertIn('draft_mode: ' + ('mtp' if draft else 'disabled'), text)
                    self.assertIn('vision: true', text)

    def test_existing_profile_migrates_to_vision_and_explicit_opt_out_is_retained(self):
        old_fields = asdict(Profile()); old_fields.pop('vision')
        self.assertTrue(Profile(**old_fields).vision)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'config.nccl.yml').write_text((ROOT / 'config.nccl.yml').read_text())
            value = write_profile(root, replace(Profile(), vision=False))
            self.assertFalse(Profile(**value['profile']).vision)
            self.assertIn('vision: false', (root / 'config.yml').read_text())

    def test_force_retains_port_keys_and_gpu_selection(self):
        inventory = '\n'.join(f'{i}, GPU-{i:08x}-aaaa, A100 80GB, 81920, 580.95.05, Disabled' for i in range(4))
        topo = ' GPU0 GPU1 GPU2 GPU3\nGPU0 X SYS NV4 SYS\nGPU1 SYS X SYS SYS\nGPU2 NV4 SYS X SYS\nGPU3 SYS SYS SYS X'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'config.nccl.yml').write_text((ROOT / 'config.nccl.yml').read_text())
            def invoke(args):
                with patch.object(configure, 'ROOT', root), patch.object(configure, 'command', side_effect=[inventory, topo]), patch.object(sys, 'argv', ['configure.py', *args]), patch('builtins.print'):
                    configure.main()
            invoke(['--gpus', '0,1,2', '--port', '5005'])
            old = (root / 'secrets/api_tokens.yml').read_bytes()
            invoke(['--force', '--mode', 'layer'])
            self.assertEqual(old, (root / 'secrets/api_tokens.yml').read_bytes())
            self.assertIn('API_PORT=5005', (root / '.env').read_text())
            self.assertIn('GPU1_UUID=GPU-00000002-aaaa', (root / '.env').read_text())
            self.assertEqual((root / 'secrets/api_tokens.yml').stat().st_mode & 0o777, 0o600)


class TopologyTests(unittest.TestCase):
    def topology(self, indices, pair):
        text = ' '.join(f'GPU{i}' for i in indices) + ' NIC0 CPU Affinity NUMA Affinity\n'
        for i in indices:
            cells = ['X' if i == j else 'NV12' if {i, j} == set(pair) else 'PHB' for j in indices]
            text += f'GPU{i} ' + ' '.join(cells) + ' PIX 0-15 0\n'
        return text

    def test_all_pair_positions_and_noncontiguous_indices(self):
        for indices in [(0, 1, 2), (2, 5, 8)]:
            for pair in [(indices[0], indices[1]), (indices[0], indices[2]), (indices[1], indices[2])]:
                topo = self.topology(indices, pair)
                with self.subTest(indices=indices, pair=pair):
                    self.assertEqual(nvlink_pairs(topo), [pair])
                    inventory = '\n'.join(f'{i}, GPU-uuid-{i}' for i in reversed(indices))
                    selected = [f'GPU-uuid-{i}' for i in reversed(pair)]
                    self.assertEqual(require_nvlink_pair(topo, inventory, selected), pair)

    def test_missing_asymmetric_and_zero_links_are_rejected(self):
        text = self.topology((0, 1, 2), (0, 1))
        for broken in [text.replace('NV12', 'PHB'), text.replace('NV12', 'NV0'), text.replace('NV12', 'PHB', 1)]:
            self.assertEqual(nvlink_pairs(broken), [])
            with self.assertRaisesRegex(RuntimeError, 'NVLink connection'):
                require_nvlink_pair(broken, '0, GPU-a\n1, GPU-b\n2, GPU-c', ['GPU-a', 'GPU-b'])

    def test_inventory_mapping_prevents_accepting_the_wrong_pair(self):
        with self.assertRaisesRegex(RuntimeError, 'NVLink connection'):
            require_nvlink_pair(self.topology((0, 1, 2), (0, 1)),
                                '0, GPU-a\n1, GPU-b\n2, GPU-c', ['GPU-a', 'GPU-c'])


class RankingTests(unittest.TestCase):
    def test_capacity_alone_cannot_win(self):
        slow, fast, failed = record('native'), record('nccl', 50, 40, 2, .2), record('layer', 1, 1000, .1, .01, 'failed')
        self.assertEqual([r['name'] for r in autotune.rank_profiles([slow, fast, failed])], ['nccl', 'native'])

    def test_larger_context_rejected_if_slower(self):
        self.assertFalse(autotune.no_material_regression(record('native', speed=10), record('nccl'), .05))
        self.assertTrue(autotune.no_material_regression(record('native', speed=20.1), record('nccl'), .05))

    def test_failed_fastest_boundary_falls_back(self):
        tuner = autotune.Tuner.__new__(autotune.Tuner)
        tuner.progress = Mock()
        tuner.args = argparse.Namespace(modes=['nccl', 'native'], chunks=[2048], draft_tokens=[0], no_expand=True)
        tuner.report = {'records': []}; tuner.root = ROOT; tuner.directory = ROOT / 'reports/test'
        def evaluate(profile):
            value = record(profile.mode, speed=40 if profile.mode == 'nccl' else 20)
            value['elapsed_s'] = 10
            tuner.report['records'].append(value)
            return value
        tuner.evaluate = evaluate; tuner.stop = Mock(); tuner.save = Mock(); tuner.start = Mock()
        tuner.check = Mock(); tuner.restore = Mock()
        tuner.boundary = Mock(side_effect=[RuntimeError('OOM near ceiling'), {'ok': True}])
        with patch.object(autotune, 'write_profile'), patch('builtins.print'):
            tuner.optimize(Profile())
        self.assertEqual(tuner.report['selected']['mode'], 'native')
        tuner.restore.assert_not_called()

    def test_interruption_restores_previous_service(self):
        tuner = autotune.Tuner.__new__(autotune.Tuner)
        tuner.progress = Mock()
        tuner.args = argparse.Namespace(modes=['nccl'], chunks=[2048], draft_tokens=[0])
        tuner.report = {'records': []}; tuner.evaluate = Mock(side_effect=KeyboardInterrupt)
        tuner.save = Mock(); tuner.restore = Mock()
        with self.assertRaises(KeyboardInterrupt):
            tuner.optimize(Profile())
        tuner.restore.assert_called_once()
        self.assertEqual(tuner.report['status'], 'failed-original-restored')

    def test_restore_preserves_original_config_and_stopped_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            tuner = autotune.Tuner.__new__(autotune.Tuner)
            tuner.progress = Mock()
            tuner.root = Path(tmp); tuner.stop = Mock(); tuner.wait_healthy = Mock()
            tuner.was_running = False; tuner.originals = {'config.yml': b'original: value\n', 'deployment.json': None}
            (tuner.root / 'config.yml').write_text('candidate')
            (tuner.root / 'deployment.json').write_text('{}')
            with patch.object(autotune, 'compose') as compose:
                tuner.restore(); compose.assert_not_called()
            self.assertEqual((tuner.root / 'config.yml').read_bytes(), b'original: value\n')
            self.assertFalse((tuner.root / 'deployment.json').exists())


class BenchmarkAssertions(unittest.TestCase):
    def item(self, number, start, end, count=4096):
        return {'request': number, 'usage': {'prompt_tokens': 260000, 'completion_tokens': count},
                'first_output_s': start, 'last_output_s': end, 'finished_s': end+.1, 'started_s': 0,
                'ttft_s': start, 'decode_tokens_per_s': count/(end-start), 'stream_events': 20,
                'p95_stream_gap_s': .1, 'max_stream_gap_s': .2}

    def test_serial_one_token_responses_fail(self):
        _, errors = summarize([self.item(1, 1, 2, 1), self.item(2, 3, 4, 1)], [260000]*2, [4096]*2)
        self.assertTrue(any('insufficient output' in e for e in errors))
        self.assertTrue(any('overlapping' in e for e in errors))

    def test_truncated_prompt_fails(self):
        _, errors = summarize([self.item(1, 1, 5), self.item(2, 2, 6)], [261000]*2, [4096]*2)
        self.assertTrue(any('input changed' in e for e in errors))

    def test_extra_session_must_join_while_both_long_sessions_generate(self):
        results = [self.item(1, 1, 10), self.item(2, 2, 11), self.item(3, 12, 13)]
        _, errors = summarize(results, [260000]*3, [4096]*3, mixed=True)
        self.assertTrue(any('extra session' in e for e in errors))

    def test_true_overlap_passes(self):
        _, errors = summarize([self.item(1, 1, 10), self.item(2, 2, 11), self.item(3, 3, 4)],
                              [260000]*3, [4096]*3, mixed=True)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
