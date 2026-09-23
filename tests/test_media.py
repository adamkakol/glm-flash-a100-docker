import base64
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import media_chat
from scripts.vision_input import solid_image


class MediaTests(unittest.TestCase):
    def test_sampling_is_bounded_and_covers_full_clip(self):
        times = media_chat.sample_times(3600, 2, 16)
        self.assertEqual(len(times), 16)
        self.assertEqual(times[0], 0)
        self.assertGreater(times[-1], 3300)
        self.assertEqual(media_chat.sample_times(1, 2, 16), [0, .5])
        self.assertEqual(media_chat.sample_times(1.1, 2, 16), [0, .5, 1.0])
        for duration in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                media_chat.sample_times(duration, 2, 16)

    def test_image_is_embedded_without_remote_fetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'sample.png'
            data = base64.b64decode(solid_image('blue', size=64).split(',', 1)[1])
            image.write_bytes(data)
            parts, metadata = media_chat.media_content(image, 'image')
            self.assertEqual(base64.b64decode(parts[0]['image_url']['url'].split(',', 1)[1]), data)
            self.assertEqual(metadata['input'], 'image')
            self.assertTrue(image.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools unavailable')
    def test_real_synthetic_video_extracts_timestamped_frames_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.mkv'
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                            'testsrc2=size=64x48:rate=4:duration=1', '-c:v', 'ffv1', str(path)], check=True)
            parts, metadata = media_chat.media_content(path, 'video', fps=4, max_frames=3, max_side=64)
            images = [p for p in parts if p['type'] == 'image_url']
            self.assertEqual(len(images), 3)
            self.assertEqual(len(metadata['sampled_timestamps_s']), 3)
            self.assertFalse(metadata['native_video'])
            self.assertFalse(metadata['audio_included'])
            self.assertTrue(all(base64.b64decode(p['image_url']['url'].split(',', 1)[1]).startswith(b'\xff\xd8') for p in images))
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_missing_video_duration_is_rejected_before_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.mkv'; path.touch()
            with patch.object(media_chat, 'command', return_value=b'{"format": {}}') as command:
                with self.assertRaisesRegex(ValueError, 'duration'):
                    media_chat.media_content(path, 'video')
                command.assert_called_once()


if __name__ == '__main__':
    unittest.main()
