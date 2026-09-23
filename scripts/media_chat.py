#!/usr/bin/env python3
"""Ask about an image or timestamped video frames using the existing image API."""
import argparse
import base64
import json
import math
from pathlib import Path
import subprocess
import sys

try:
    from .api_check import Client
    from .deployment import MODEL_NAME, read_env
except ImportError:
    from api_check import Client
    from deployment import MODEL_NAME, read_env

ROOT = Path(__file__).resolve().parents[1]
MAX_MEDIA_BYTES = 32 * 1024 * 1024
IMAGE_TYPES = {'.png': 'png', '.jpg': 'jpeg', '.jpeg': 'jpeg', '.webp': 'webp', '.bmp': 'bmp'}


def data_url(data, kind):
    return 'data:image/' + kind + ';base64,' + base64.b64encode(data).decode('ascii')


def sample_times(duration, fps, max_frames):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Video must have a finite, positive duration')
    if not math.isfinite(fps) or not 0 < fps <= 10 or not 1 <= max_frames <= 64:
        raise ValueError('Use 0 < fps <= 10 and 1–64 frames')
    count = min(max_frames, max(1, math.ceil(duration * fps)))
    interval = max(1 / fps, duration / count)
    return [i * interval for i in range(count)]


def command(args, timeout=120):
    result = subprocess.run(args, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr[-1200:].decode(errors="replace")}')
    return result.stdout


def media_content(path, kind, *, fps=2, max_frames=16, max_side=1024):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Media path must name a local file')
    if kind == 'image':
        suffix = IMAGE_TYPES.get(path.suffix.lower())
        if not suffix:
            raise ValueError('Use PNG, JPEG, WebP or BMP for image mode')
        if path.stat().st_size > MAX_MEDIA_BYTES:
            raise ValueError('Image exceeds the 32 MiB upload budget')
        return ([{'type': 'image_url', 'image_url': {'url': data_url(path.read_bytes(), suffix)}}],
                {'input': 'image'})
    if kind != 'video' or not 64 <= max_side <= 2048:
        raise ValueError('Use video mode and 64–2048 pixels for max-side')
    info = json.loads(command(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                               '-of', 'json', str(path)]))
    duration = float(info.get('format', {}).get('duration', 'nan'))
    times = sample_times(duration, fps, max_frames)
    parts = [{'type': 'text', 'text': (
        f'These are {len(times)} sampled frames spanning a {duration:.3f}-second video. '
        'Timestamps are approximate. The audio track is not supplied. '
        'Describe only evidence in the sampled frames; fast events between them may be missing.')}]
    total = 0
    scale = f"scale=w='min({max_side},iw)':h='min({max_side},ih)':force_original_aspect_ratio=decrease"
    for timestamp in times:
        frame = command(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{timestamp:.6f}', '-i', str(path),
                         '-map', '0:v:0', '-frames:v', '1', '-an', '-sn', '-vf', scale,
                         '-c:v', 'mjpeg', '-q:v', '3', '-f', 'image2pipe', 'pipe:1'])
        if not frame.startswith(b'\xff\xd8'):
            raise RuntimeError(f'No decodable frame near {timestamp:.3f} seconds')
        total += len(frame)
        if total > MAX_MEDIA_BYTES:
            raise ValueError('Frames exceed 32 MiB; reduce --max-frames or --max-side')
        parts += [{'type': 'text', 'text': f'\nFrame near {timestamp:.3f} seconds:\n'},
                  {'type': 'image_url', 'image_url': {'url': data_url(frame, 'jpeg')}}]
    return parts, {'input': 'video-sampled-frames', 'duration_s': duration,
                   'sampled_timestamps_s': times, 'native_video': False, 'audio_included': False}


def ask(args, root=ROOT):
    env = read_env(root)
    keys = json.loads((root / 'secrets/api_tokens.yml').read_text())['api_key']
    key = keys[0] if isinstance(keys, list) else keys
    client = Client(args.url or f'http://127.0.0.1:{env.get("API_PORT", "5000")}', key,
                    timeout=args.timeout, reasoning_effort=args.reasoning_effort)
    model = client.json('/v1/model')
    if model['id'] != MODEL_NAME or model['parameters'].get('use_vision') is not True:
        raise RuntimeError('Expected the configured GLM model with vision enabled')
    content, metadata = media_content(args.path, args.kind, fps=args.fps,
                                     max_frames=args.max_frames, max_side=args.max_side)
    content.append({'type': 'text', 'text': '\n' + args.prompt})
    messages = [{'role': 'user', 'content': content}]
    client.calibrate_prompt_count(model['id'])
    count = client.token_count(messages)
    params = model['parameters']
    if count + args.max_output + 2048 > min(params['max_seq_len'], params['cache_size']):
        raise RuntimeError('Media plus output exceed the configured window; reduce frames, image size or output budget')
    result = client.json('/v1/chat/completions', {'model': model['id'], 'messages': messages,
        'max_tokens': args.max_output, 'temperature': 1.0, 'top_p': .95,
        'chat_template_kwargs': {'reasoning_effort': args.reasoning_effort, 'clear_thinking': True}})
    actual = result.get('usage', {}).get('prompt_tokens')
    if actual is None or abs(actual - count) > 2:
        raise RuntimeError(f'Input token count changed from {count} to {actual}')
    return {'media': metadata, 'prompt_tokens': count, 'response': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['image', 'video'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--prompt', default='Describe the supplied visual material in detail.')
    parser.add_argument('--url')
    parser.add_argument('--fps', type=float, default=2, help='Maximum sampling rate; frame cap spreads samples across the full clip')
    parser.add_argument('--max-frames', type=int, default=16)
    parser.add_argument('--max-side', type=int, default=1024, help='Maximum video-frame width/height')
    parser.add_argument('--max-output', type=int, default=8192)
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='max')
    args = parser.parse_args()
    if args.max_output < 1 or args.timeout < 1:
        parser.error('Output and timeout must be positive')
    if not math.isfinite(args.fps) or not 0 < args.fps <= 10 or not 1 <= args.max_frames <= 64 or not 64 <= args.max_side <= 2048:
        parser.error('Use 0 < fps <= 10, 1–64 frames and 64–2048 pixels for max-side')
    try:
        print(json.dumps(ask(args), indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Media request failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
