"""Small in-memory PNG fixtures and OpenAI image message construction."""
import base64
import struct
import uuid
import zlib

COLORS = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0)}


def solid_image(color, *, size=1024, nonce=None):
    """Unique URLs bypass Tabby's embedding cache while preserving identical pixels."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = (b'\x00' + bytes(COLORS[color]) * size) * size
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
           + chunk(b'tEXt', b'Request\x00' + (nonce or uuid.uuid4().hex).encode('ascii'))
           + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')


def image_content(text, url):
    return [{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': text}]
