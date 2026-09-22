"""Media helpers (standard library only): wrap JPEG frames into an MJPEG AVI, write WAV.

A browser can capture camera frames as JPEGs but can't easily produce an MP4. vLLM's
video loader reads MJPEG AVI, so systemone accepts `video_frames` (a list of JPEG data
URLs) and packs them into a clip server-side.
"""

import base64
import io
import struct
import wave
from typing import List


# a 32x32 grey JPEG, used to probe whether a model takes video
PROBE_JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwAooooAKKKKACiiigAooooA/9k="


def data_url_bytes(url: str) -> bytes:
    return base64.b64decode(url.split(",", 1)[1])


def _jpeg_size(jpg: bytes):
    """(width, height) from a JPEG's SOF marker."""
    i = 2
    while i < len(jpg):
        if jpg[i] != 0xFF:
            i += 1
            continue
        marker = jpg[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", jpg[i + 5:i + 9])
            return w, h
        seg = struct.unpack(">H", jpg[i + 2:i + 4])[0]
        i += 2 + seg
    raise ValueError("not a baseline/progressive JPEG")


def mjpeg_avi(jpegs: List[bytes], fps: float) -> bytes:
    """A minimal AVI (RIFF, one MJPG video stream, idx1 index) from JPEG frames."""
    if not jpegs:
        raise ValueError("no frames")
    w, h = _jpeg_size(jpegs[0])
    n = len(jpegs)
    us = int(1_000_000 / fps)

    def chunk(fourcc, data):
        pad = b"\0" if len(data) % 2 else b""
        return fourcc + struct.pack("<I", len(data)) + data + pad

    def lst(kind, body):
        return b"LIST" + struct.pack("<I", len(body) + 4) + kind + body

    avih = struct.pack("<IIIIIIIIII4I", us, max(len(j) for j in jpegs) * int(fps + 1), 0, 0x10, n, 0, 1,
                       max(len(j) for j in jpegs), w, h, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"MJPG", 0, 0, 0, 0, 1000, int(fps * 1000), 0, n,
                       max(len(j) for j in jpegs), 0xFFFFFFFF, 0, 0, 0, w, h)
    strf = struct.pack("<IiiHH4sIiiII", 40, w, h, 1, 24, b"MJPG", w * h * 3, 0, 0, 0, 0)
    hdrl = lst(b"hdrl", chunk(b"avih", avih) + lst(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi_body, index, offset = b"", b"", 4
    for j in jpegs:
        c = chunk(b"00dc", j)
        index += b"00dc" + struct.pack("<III", 0x10, offset, len(j))
        movi_body += c
        offset += len(c)
    movi = lst(b"movi", movi_body)
    body = b"AVI " + hdrl + movi + chunk(b"idx1", index)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def frames_to_video_url(frame_urls: List[str], fps: float) -> str:
    return "data:video/x-msvideo;base64," + base64.b64encode(mjpeg_avi([data_url_bytes(u) for u in frame_urls], fps)).decode()


def wav_url(samples: List[int], rate: int = 16000) -> str:
    """PCM16 mono samples as a WAV data URL."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
