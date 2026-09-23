import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import autotune
from scripts.deployment import Profile
from scripts.progress import BenchmarkProgress, TuningProgress


class ProgressTests(unittest.TestCase):
    def test_bar_and_eta_do_not_claim_full_run_completion(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            progress = TuningProgress(tmp, total=4)
            progress.set(phase='matrix', stage='long test', candidate='fixture', index=1)
            self.assertIsNone(progress.snapshot()['matrix_eta_s'])
            progress.complete_candidate(True, 10)
            progress.set(index=2)
            progress.complete_candidate(True, 20)
            value = progress.snapshot()
            self.assertAlmostEqual(value['matrix_eta_s'], 30, delta=.1)
            self.assertIn('2/4 (50%)', progress.render(value))
            progress.complete_candidate(False, 1)
            progress.complete_candidate(True, 20)
            progress.set(phase='context boundary', stage='Checking maximum context')
            value = json.loads((Path(tmp) / 'progress.json').read_text())
            self.assertEqual(value['status'], 'running')
            self.assertEqual(value['matrix_completed'], 4)
            self.assertIsNone(value['matrix_eta_s'])
            self.assertIn('context boundary', progress.render(value))
            progress.set(phase='complete', status='selected', stage='Selected profile running')
            self.assertEqual(progress.snapshot()['status'], 'selected')

    def test_heartbeat_continues_without_stage_or_output_events(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with TuningProgress(tmp, total=1, interval=.02) as progress:
                progress.set(stage='Waiting for first output')
                threading.Event().wait(.12)
            text = (Path(tmp) / 'progress.log').read_text()
            self.assertGreaterEqual(text.count('Waiting for first output'), 3)
            self.assertFalse(progress.thread.is_alive())
            value = json.loads((Path(tmp) / 'progress.json').read_text())
            self.assertGreater(value['stage_elapsed_s'], .05)

    def test_stream_events_are_not_presented_as_generated_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'activity.json'
            with BenchmarkProgress(path, interval=.02) as activity:
                activity.stage('Streaming')
                activity.request(1, status='waiting for first output', input_tokens=260000)
                activity.output(1); activity.output(1)
                activity.publish()
                value = json.loads(path.read_text())
                self.assertEqual(value['requests']['1']['output_events'], 2)
                self.assertNotIn('completion_tokens', value['requests']['1'])
                activity.request(1, status='completed', completion_tokens=4096)
                activity.finish('passed')
            self.assertEqual(json.loads(path.read_text())['requests']['1']['completion_tokens'], 4096)
            self.assertFalse(activity.thread.is_alive())

    def test_stale_snapshot_is_identified_and_partial_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = TuningProgress(tmp, total=1)
            path = Path(tmp) / 'activity.json'
            path.write_text('{')
            progress.follow(path)
            self.assertIsNone(progress.snapshot()['benchmark'])
            path.write_text(json.dumps({'stage': 'Streaming', 'updated_unix_s': 100,
                'elapsed_s': 10, 'requests': {'1': {'status': 'generating', 'output_events': 2,
                                                'last_output_elapsed_s': 8}}}))
            with patch('scripts.progress.time.time', return_value=130):
                text = progress.render(progress.snapshot())
            self.assertIn('snapshot age 30s', text)
            self.assertIn('last output 32s ago', text)


class ProcessTests(unittest.TestCase):
    def test_real_timeout_terminates_and_reaps_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'child.log'
            command = [sys.executable, '-u', '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(30)']
            with self.assertRaises(subprocess.TimeoutExpired):
                autotune.run_logged(command, root=tmp, log=log, timeout=.2)
            pid = int(log.read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_keyboard_interrupt_terminates_child_before_propagating(self):
        process = Mock(); process.wait.side_effect = [KeyboardInterrupt, 0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(autotune.subprocess, 'Popen', return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                autotune.run_logged(['fixture'], root=tmp, log=Path(tmp) / 'child.log', timeout=10)
        process.terminate.assert_called_once()
        process.kill.assert_not_called()

    def test_unresponsive_child_is_killed(self):
        process = Mock(); process.wait.side_effect = [subprocess.TimeoutExpired('fixture', 1),
                                                      subprocess.TimeoutExpired('fixture', 5), 0]
        with tempfile.TemporaryDirectory() as tmp, patch.object(autotune.subprocess, 'Popen', return_value=process):
            with self.assertRaises(subprocess.TimeoutExpired):
                autotune.run_logged(['fixture'], root=tmp, log=Path(tmp) / 'child.log', timeout=1)
        process.terminate.assert_called_once(); process.kill.assert_called_once()

    def test_tuner_displays_child_activity_and_completed_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(tmp); (root / 'scripts').mkdir(); reports = root / 'reports'; reports.mkdir()
            # A real child process simulates a quiet prefill, then output; no Docker/GPU/model.
            (root / 'scripts/api_check.py').write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--report');p.add_argument('--progress-json');a,_=p.parse_known_args()
x={'stage':'Streaming fixture','status':'running','updated_unix_s':time.time(),'elapsed_s':0,
   'requests':{'1':{'status':'waiting for first output'}}}
Path(a.progress_json).write_text(json.dumps(x));time.sleep(.12)
x['updated_unix_s']=time.time();x['elapsed_s']=.12
x['requests']['1']={'status':'generating','output_events':3,'last_output_elapsed_s':.12}
Path(a.progress_json).write_text(json.dumps(x));time.sleep(.12)
r={'status':'passed','metrics':{'worst_ttft_s':7,'worst_decode_tokens_per_s':20,
   'worst_max_stream_gap_s':.2,'generation_overlap_fraction':.9}}
Path(a.report).write_text(json.dumps(r));print('fixture child completed',flush=True)
''')
            tuner = autotune.Tuner.__new__(autotune.Tuner)
            tuner.root = root; tuner.directory = reports
            tuner.args = argparse.Namespace(tokens=260000, output_tokens=4096, request_timeout=10,
                reasoning_effort='max', image_size=1024, corpus=None, repeats=2)
            tuner.progress = TuningProgress(reports, total=1, interval=.02)
            with tuner.progress:
                report = tuner.check(Profile(), 'long', 0)
            text = output.getvalue()
            self.assertIn('waiting for first output', text)
            self.assertIn('3 output events', text)
            self.assertIn('20.00 tokens/s', text)
            self.assertEqual(report['status'], 'passed')
            log = reports / f'{Profile().name}-0-long.log'
            self.assertIn('fixture child completed', log.read_text())
            state = json.loads((reports / 'progress.json').read_text())
            self.assertEqual(state['last_metrics']['worst_ttft_s'], 7)


if __name__ == '__main__':
    unittest.main()
