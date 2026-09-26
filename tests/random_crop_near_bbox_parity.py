#!/usr/bin/env python3
"""Reference checks for the explicit RandomCropNearBBox contract."""
import math
import pathlib
import struct
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1

def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) & MASK

def f32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]

def rectangle(width, height, box, shifts, seed, explicit=None):
    x1, y1, x2, y2 = box
    max_x = int((f32(shifts[0]) * (x2 - x1)) // 1)
    max_y = int((f32(shifts[1]) * (y2 - y1)) // 1)
    if explicit is None:
        explicit = (-1, -1, -1, -1)
    offsets = tuple(value if value >= 0 else splitmix64(seed + side) % (limit + 1)
                    for side, (value, limit) in enumerate(zip(explicit, (max_x, max_x, max_y, max_y))))
    left, right, top, bottom = offsets
    cx0 = max(0, int(x1) - left)
    cy0 = max(0, int(y1) - top)
    cx1 = min(width, int(math.ceil(x2)) + right)
    cy1 = min(height, int(math.ceil(y2)) + bottom)
    return cx0, cy0, cx1 - cx0, cy1 - cy0

def expected(image, width, height, channels, box, shifts, seed, explicit=None):
    x, y, w, h = rectangle(width, height, box, shifts, seed, explicit)
    out = bytearray()
    for row in range(y, y + h):
        out.extend(image[(row * width + x) * channels:(row * width + x + w) * channels])
    return bytes(out)

def run(exe, image, width, height, channels, box, shifts, seed, td, explicit=None):
    source = pathlib.Path(td) / "in.raw"
    target = pathlib.Path(td) / "out.raw"
    source.write_bytes(image)
    args = [exe, "random_crop_near_bbox", str(source), str(target), str(width), str(height), str(channels)]
    args += [str(value) for value in box] + [str(value) for value in shifts] + [str(seed)]
    if explicit is not None:
        args += [str(value) for value in explicit]
    subprocess.run(args, check=True)
    return target.read_bytes()

def main():
    exe = sys.argv[1]
    width, height, channels = 13, 10, 2
    image = bytes((index * 29 + 11) & 255 for index in range(width * height * channels))
    box, shifts = (3, 2, 9, 7), (0.5, 0.4)
    with tempfile.TemporaryDirectory() as td:
        for seed in (0, 1, 77, 0x123456789ABCDEF0):
            actual = run(exe, image, width, height, channels, box, shifts, seed, td)
            assert actual == expected(image, width, height, channels, box, shifts, seed)
            assert run(exe, image, width, height, channels, box, shifts, seed, td) == actual
        explicit = (1, 2, 1, 0)
        assert run(exe, image, width, height, channels, box, shifts, 999, td, explicit) == expected(
            image, width, height, channels, box, shifts, 999, explicit)
        fractional_box = (3.2, 2.4, 8.7, 6.1)
        fractional_shifts = (0.5, 0.4)
        fractional_offsets = (1, 2, 1, 0)
        assert run(exe, image, width, height, channels, fractional_box, fractional_shifts, 999, td, fractional_offsets) == expected(
            image, width, height, channels, fractional_box, fractional_shifts, 999, fractional_offsets)
        identity = (0.0, 0.0)
        assert run(exe, image, width, height, channels, box, identity, 42, td) == expected(
            image, width, height, channels, box, identity, 42)
        bad = subprocess.run(
            [exe, "random_crop_near_bbox", str(pathlib.Path(td) / "in.raw"), str(pathlib.Path(td) / "bad.raw"),
             str(width), str(height), str(channels), "3", "2", "9", "7", "0.5", "0.4", "1", "4", "0", "0", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert bad.returncode != 0
    print("PASS: RandomCropNearBBox exact deterministic crop, explicit offsets, and validation")

if __name__ == "__main__":
    main()
