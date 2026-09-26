#!/usr/bin/env python3
"""Reference parity for the explicit GlassBlur swap-sequence/seed contract."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def gaussian(image, sigma):
    radius = int(np.ceil(3.0 * sigma))
    height, width, channels = image.shape
    result = np.empty_like(image)
    weights = np.empty((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weights[dy + radius, dx + radius] = np.exp(
                np.float32(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            )
    norm = np.sum(weights, dtype=np.float32)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                total = np.float32(0.0)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        py = abs(y + dy) if y + dy < 0 else y + dy
                        px = abs(x + dx) if x + dx < 0 else x + dx
                        if py >= height:
                            py = 2 * height - py - 2
                        if px >= width:
                            px = 2 * width - px - 2
                        total += weights[dy + radius, dx + radius] * image[py, px, channel]
                result[y, x, channel] = np.uint8(np.floor(total / norm + np.float32(0.5)))
    return result


def reference(image, sigma, max_delta, iterations, swaps):
    work = gaussian(image, sigma)
    height, width, _ = image.shape
    index = 0
    for _ in range(iterations):
        for y in range(max_delta, height - max_delta):
            for x in range(max_delta, width - max_delta):
                dx, dy = swaps[index]
                index += 1
                work[y, x, :], work[y + dy, x + dx, :] = (
                    work[y + dy, x + dx, :].copy(),
                    work[y, x, :].copy(),
                )
    return gaussian(work, sigma)


def seeded_swaps(width, height, max_delta, iterations, seed):
    count = iterations * (height - 2 * max_delta) * (width - 2 * max_delta)
    span = 2 * max_delta + 1
    return [
        (
            splitmix64(seed + 2 * i) % span - max_delta,
            splitmix64(seed + 2 * i + 1) % span - max_delta,
        )
        for i in range(count)
    ]


exe = sys.argv[1]
width, height, channels = 9, 8, 2
sigma, max_delta, iterations, seed = 0.7, 1, 2, 0x123456789ABCDEF0
image = np.arange(width * height * channels, dtype=np.uint8).reshape(height, width, channels)
count = iterations * (height - 2 * max_delta) * (width - 2 * max_delta)
explicit = [((i % 3) - 1, ((i * 2) % 3) - 1) for i in range(count)]
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source, output, repeat = root / "input.raw", root / "output.raw", root / "repeat.raw"
    source.write_bytes(image.tobytes())
    command = [exe, "glass_blur", str(source), str(output), str(width), str(height), str(channels),
               str(sigma), str(max_delta), str(iterations), str(seed)]
    subprocess.run(command, check=True)
    expected = reference(image, sigma, max_delta, iterations, seeded_swaps(width, height, max_delta, iterations, seed))
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.uint8).reshape(image.shape), expected)
    repeat_command = command.copy()
    repeat_command[3] = str(repeat)
    subprocess.run(repeat_command, check=True)
    np.testing.assert_array_equal(output.read_bytes(), repeat.read_bytes())
    sequence = root / "swaps.i8"
    sequence.write_bytes(bytes(value & 0xFF for pair in explicit for value in pair))
    subprocess.run(command + [str(sequence)], check=True)
    expected_explicit = reference(image, sigma, max_delta, iterations, explicit)
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.uint8).reshape(image.shape), expected_explicit)
print("PASS: GlassBlur swap-sequence and seed reference parity")
