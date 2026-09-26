#!/usr/bin/env python3
"""Exact and property checks for deterministic RandomCropFromBorders."""
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

def offsets(width, height, fractions, seed, explicit=None):
    limits = [int(f32(f) * size) for f, size in zip(fractions, (width, width, height, height))]
    if explicit is None:
        explicit = (-1, -1, -1, -1)
    result = tuple(value if value >= 0 else splitmix64(seed + side) % (limit + 1)
                   for side, (value, limit) in enumerate(zip(explicit, limits)))
    return result

def expected(image, width, height, channels, fractions, seed, explicit=None):
    left, right, top, bottom = offsets(width, height, fractions, seed, explicit)
    out_width, out_height = width - left - right, height - top - bottom
    result = bytearray()
    for y in range(top, top + out_height):
        begin = (y * width + left) * channels
        result.extend(image[begin:begin + out_width * channels])
    return bytes(result), (left, right, top, bottom)

def run(exe, image, width, height, channels, fractions, seed, td, explicit=None):
    source = pathlib.Path(td) / "in.raw"
    target = pathlib.Path(td) / "out.raw"
    source.write_bytes(image)
    args = [exe, "crop_from_borders", str(source), str(target), str(width), str(height), str(channels)]
    args += [str(value) for value in fractions] + [str(seed)]
    if explicit is not None:
        args += [str(value) for value in explicit]
    subprocess.run(args, check=True)
    return target.read_bytes()

def main():
    exe = sys.argv[1]
    width, height, channels = 9, 8, 3
    image = bytes((index * 17 + 3) & 255 for index in range(width * height * channels))
    fractions = (0.3, 0.2, 0.25, 0.15)
    with tempfile.TemporaryDirectory() as td:
        for seed in (0, 1, 77, 0x123456789ABCDEF0):
            actual = run(exe, image, width, height, channels, fractions, seed, td)
            reference, crop_offsets = expected(image, width, height, channels, fractions, seed)
            assert actual == reference, (seed, crop_offsets)
            assert run(exe, image, width, height, channels, fractions, seed, td) == actual
        explicit = (2, 1, 2, 1)
        assert run(exe, image, width, height, channels, fractions, 999, td, explicit) == expected(
            image, width, height, channels, fractions, 999, explicit)[0]
        identity = (0.0, 0.0, 0.0, 0.0)
        assert run(exe, image, width, height, channels, identity, 42, td) == image
        bad = subprocess.run(
            [exe, "crop_from_borders", str(pathlib.Path(td) / "in.raw"), str(pathlib.Path(td) / "bad.raw"),
             str(width), str(height), str(channels), "0.8", "0.8", "0", "0", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert bad.returncode != 0
    print("PASS: RandomCropFromBorders exact parity, deterministic seed/offset, and crop properties")

if __name__ == "__main__":
    main()
