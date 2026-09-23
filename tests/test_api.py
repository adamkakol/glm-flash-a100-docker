"""Local HTTP fixtures exercise real SSE parsing; no model or Docker runtime."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import api_check
from scripts.deployment import MODEL_NAME


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status); self.send_header('Content-Length', str(len(payload)))
        self.end_headers(); self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/health':
            self.send_json({'status': 'healthy'})
        elif self.path == '/v1/model':
            self.send_json({'id': MODEL_NAME, 'parameters': {'max_seq_len': 524288,
                'cache_size': 1048576, 'max_batch_size': 4, 'chunk_size': 2048, 'cache_mode': 'FP16'}})
        else:
            self.send_json({'error': 'not found'}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/v1/token/encode':
            count = len(body['text'][0]['content']) // 3
            self.send_json({'length': count}); return
        if not body.get('stream'):
            self.send_json({'choices': [{'message': {'content': 'READY'}}],
                'usage': {'prompt_tokens': len(body['messages'][0]['content']) // 3 + 5}}); return
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        def event(value):
            self.wfile.write(('data: ' + json.dumps(value) + '\n\n').encode()); self.wfile.flush()
        # Long requests remain active while shorter requests enter the server.
        chunks = 30 if body['max_tokens'] >= 4096 else 4
        for i in range(chunks):
            event({'choices': [{'delta': {'reasoning_content' if i % 2 else 'content': 'word '}}]})
            time.sleep(.005)
        count = len(body['messages'][0]['content']) // 3 + 5
        event({'choices': [{'delta': {}, 'finish_reason': 'length'}], 'usage': {
            'prompt_tokens': count, 'completion_tokens': body['max_tokens']}})
        self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def args(self, mode):
        return argparse.Namespace(url=f'http://127.0.0.1:{self.server.server_port}', users=4 if mode == 'mixed' else 2,
            mode=mode, tokens=1000, max_output=4096, min_output=4096, corpus=None,
            short_tokens=500, short_output=256, timeout=10, reasoning_effort='max')

    def test_streaming_and_mixed_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'secrets').mkdir()
            (root / 'secrets/api_tokens.yml').write_text(json.dumps({'api_key': ['fixture']}))
            for mode in ['smoke', 'long', 'mixed']:
                with self.subTest(mode=mode), patch('builtins.print'):
                    report = api_check.run_check(self.args(mode), root)
                    self.assertEqual(report['status'], 'passed', report.get('errors'))
                    if mode != 'smoke':
                        self.assertEqual(report['chat_template_token_offset'], 5)
                        self.assertGreater(report['metrics']['generation_overlap_fraction'], .1)
                        self.assertGreater(report['results'][0]['decode_tokens_per_s'], 0)

    def test_inconsistent_template_accounting_fails(self):
        client = api_check.Client('http://fixture', 'fixture')
        with patch.object(client, 'token_count', return_value=10), patch.object(client, 'json',
                side_effect=[{'usage': {'prompt_tokens': 15}}, {'usage': {'prompt_tokens': 17}}]):
            with self.assertRaisesRegex(RuntimeError, 'Inconsistent'):
                client.calibrate_prompt_count(MODEL_NAME)

    def test_failed_check_writes_report_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            with patch.object(sys, 'argv', ['api_check.py', 'long', '--report', str(path)]), \
                 patch.object(api_check, 'run_check', side_effect=RuntimeError('fixture failure')), patch('builtins.print'):
                self.assertEqual(api_check.main(), 1)
            self.assertEqual(json.loads(path.read_text())['status'], 'failed')


class HealthTests(unittest.TestCase):
    def test_engine_health_is_required_even_if_model_metadata_is_valid(self):
        # healthcheck uses PyYAML at runtime, supplied by the pinned ExLlama dependency.
        import healthcheck
        import io
        expected = {'model': {'model_name': MODEL_NAME, 'max_seq_len': 524288, 'cache_size': 1048576,
                    'max_batch_size': 4, 'chunk_size': 2048, 'cache_mode': 'FP16'}}
        def read(path):
            return json.dumps({'api_key': ['fixture']}) if path.name == 'api_tokens.yml' else json.dumps(expected)
        called = []
        def open_url(request, **kwargs):
            called.append(request.full_url)
            return io.BytesIO(json.dumps({'status': 'unhealthy'}).encode())
        with patch.object(Path, 'read_text', read), patch.object(healthcheck.urllib.request, 'urlopen', side_effect=open_url):
            with self.assertRaisesRegex(RuntimeError, 'unhealthy'):
                healthcheck.main()
        self.assertEqual(called, ['http://127.0.0.1:5000/health'])


if __name__ == '__main__':
    unittest.main()
