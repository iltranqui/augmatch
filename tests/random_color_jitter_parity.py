#!/usr/bin/env python3
"""Fixed-seed contract, native ColorJitter composition, and range properties."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

exe = sys.argv[1]
width, height, channels = 11, 7, 4
image = np.random.default_rng(2024).integers(0, 256, (height, width, channels), dtype=np.uint8)
ranges = (-0.15, 0.20, 0.72, 1.31, 0.45, 1.40, -21.0, 24.0)
seed = 987654321

def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)

def sampled(lo, hi, index):
    unit = np.float32((splitmix64(seed + index) >> 40) / float(1 << 24))
    return np.float32(np.float32(lo) + np.float32(np.float32(hi - lo) * unit))

params = [sampled(ranges[i], ranges[i + 1], i // 2) for i in range(0, 8, 2)]
# The CLI's explicit ColorJitter path is the native kernel used by the random API.
random_args = [str(v) for v in ranges] + [str(seed)]
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "in.raw"
    random_out = root / "random.raw"
    explicit_out = root / "explicit.raw"
    repeat_out = root / "repeat.raw"
    source.write_bytes(image.tobytes())
    command = [exe, "color", "random_color_jitter", str(source), str(random_out),
               str(width), str(height), str(channels)] + random_args
    subprocess.run(command, check=True)
    repeat = command[:]
    repeat[4] = str(repeat_out)
    subprocess.run(repeat, check=True)
    got = np.fromfile(random_out, dtype=np.uint8)
    assert np.array_equal(got, np.fromfile(repeat_out, dtype=np.uint8)), "fixed seed is not repeatable"
    explicit = [exe, "color", "color_jitter", str(source), str(explicit_out),
                str(width), str(height), str(channels)] + [format(float(v), ".9g") for v in params]
    subprocess.run(explicit, check=True)
    assert np.array_equal(got, np.fromfile(explicit_out, dtype=np.uint8)), "seeded parameters do not compose native ColorJitter"
    assert np.array_equal(got.reshape(image.shape)[..., 3], image[..., 3]), "extra channels must be copied"

# Degenerate ranges are an exact fixed ColorJitter configuration, independent of seed.
fixed = [0.1, 0.1, 1.2, 1.2, 0.8, 0.8, 7.0, 7.0]
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "in.raw"
    random_out = root / "random.raw"
    explicit_out = root / "explicit.raw"
    source.write_bytes(image.tobytes())
    subprocess.run([exe, "color", "random_color_jitter", str(source), str(random_out),
                    str(width), str(height), str(channels)] + [str(v) for v in fixed] + ["1"], check=True)
    subprocess.run([exe, "color", "color_jitter", str(source), str(explicit_out),
                    str(width), str(height), str(channels)] + [str(v) for v in fixed[::2]], check=True)
    assert random_out.read_bytes() == explicit_out.read_bytes(), "degenerate ranges changed fixed jitter"

print("PASS: RandomColorJitter fixed-seed generation, native composition, and properties")
