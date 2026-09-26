#!/usr/bin/env python3
"""Deterministic parity for the explicit per-pixel ElasticTransform field."""
import pathlib
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
w, h, channels = 7, 6, 2
image = bytes((i * 37 + 11) % 251 for i in range(w * h * channels))
dx = [((x - 3) * 0.17 + (y % 2) * 0.08) for y in range(h) for x in range(w)]
dy = [((y - 2) * -0.13 + (x % 3) * 0.06) for y in range(h) for x in range(w)]
fill = 19

def sample(px, py, ch):
    if 0 <= px < w and 0 <= py < h:
        return image[(py * w + px) * channels + ch]
    return fill

def reference():
    result = bytearray(w * h * channels)
    for y in range(h):
        for x in range(w):
            sx, sy = x + dx[y * w + x], y + dy[y * w + x]
            x0, y0 = int(sx // 1), int(sy // 1)
            ax, ay = sx - x0, sy - y0
            for ch in range(channels):
                value = ((1 - ay) * ((1 - ax) * sample(x0, y0, ch) + ax * sample(x0 + 1, y0, ch)) +
                         ay * ((1 - ax) * sample(x0, y0 + 1, ch) + ax * sample(x0 + 1, y0 + 1, ch)))
                result[(y * w + x) * channels + ch] = max(0, min(255, int(value + 0.5)))
    return result

def floats(values):
    return b"".join(struct.pack("<f", value) for value in values)

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source, output = root / "input.raw", root / "output.raw"
    xfield, yfield = root / "dx.f32", root / "dy.f32"
    source.write_bytes(image)
    xfield.write_bytes(floats(dx))
    yfield.write_bytes(floats(dy))
    subprocess.run([exe, "elastic_transform", str(source), str(output), str(w), str(h),
                    str(channels), str(xfield), str(yfield), str(fill)], check=True)
    actual = output.read_bytes()
expected = reference()
max_error = max(abs(a - b) for a, b in zip(actual, expected))
if max_error > 1:
    raise AssertionError(f"ElasticTransform deterministic reference error {max_error} > 1")
print(f"PASS: ElasticTransform deterministic per-pixel displacement reference parity (max error {max_error})")
