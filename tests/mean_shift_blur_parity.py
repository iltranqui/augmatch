#!/usr/bin/env python3
"""Direct-reference and property checks for deterministic MeanShiftBlur."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


def reflect101(index, size):
    if size == 1:
        return 0
    while index < 0 or index >= size:
        index = -index if index < 0 else 2 * size - index - 2
    return index


def reference(image, spatial_radius, color_radius, iterations):
    height, width, channels = image.shape
    output = np.empty_like(image)
    limit = np.float32(color_radius * color_radius)
    for y in range(height):
        for x in range(width):
            current = image[y, x].astype(np.float32)
            for _ in range(iterations):
                sums = np.zeros(channels, dtype=np.float32)
                count = 0
                for dy in range(-spatial_radius, spatial_radius + 1):
                    for dx in range(-spatial_radius, spatial_radius + 1):
                        sample = image[reflect101(y + dy, height), reflect101(x + dx, width)].astype(np.float32)
                        difference = sample - current
                        distance = np.sum(difference * difference, dtype=np.float32)
                        if distance <= limit:
                            count += 1
                            sums = np.float32(sums + sample)
                if count:
                    current = sums / np.float32(count)
            output[y, x] = np.floor(np.clip(current, 0, 255) + np.float32(0.5)).astype(np.uint8)
    return output


exe = sys.argv[1]
rng = np.random.default_rng(20250308)
image = rng.integers(0, 256, (7, 9, 3), dtype=np.uint8)
spatial_radius, color_radius, iterations = 2, 34.0, 3
expected = reference(image, spatial_radius, color_radius, iterations)


def run(source_image, radius, color, count, root, name):
    source = root / (name + ".in")
    destination = root / (name + ".out")
    source.write_bytes(source_image.tobytes())
    subprocess.run(
        [exe, "mean_shift_blur", str(source), str(destination), str(source_image.shape[1]),
         str(source_image.shape[0]), str(source_image.shape[2]), str(radius), str(color), str(count)],
        check=True,
    )
    return np.fromfile(destination, dtype=np.uint8).reshape(source_image.shape)


with tempfile.TemporaryDirectory() as temporary_directory:
    root = pathlib.Path(temporary_directory)
    np.testing.assert_array_equal(run(image, spatial_radius, color_radius, iterations, root, "random"), expected)
    constant = np.full((3, 4, 2), 117, dtype=np.uint8)
    np.testing.assert_array_equal(run(constant, 3, 1.0, 4, root, "constant"), constant)
    singleton = np.array([[[249]]], dtype=np.uint8)
    np.testing.assert_array_equal(run(singleton, 1, 1.0, 2, root, "singleton"), singleton)
    edge_preserving = run(image, 2, 0.01, 2, root, "edge_preserving")
    np.testing.assert_array_equal(edge_preserving, image)
    assert np.all((edge_preserving >= 0) & (edge_preserving <= 255))
print("PASS: MeanShiftBlur direct-reference parity and properties")
