#!/usr/bin/env python3
"""Direct-reference and property checks for the deterministic BilateralBlur contract."""
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


def reference(image, radius, sigma_space, sigma_color):
    height, width, channels = image.shape
    output = np.empty_like(image)
    spatial_inv = np.float32(1.0 / (2.0 * sigma_space * sigma_space))
    color_inv = np.float32(1.0 / (2.0 * sigma_color * sigma_color))
    for y in range(height):
        for x in range(width):
            center = image[y, x].astype(np.float32)
            sums = np.zeros(channels, dtype=np.float32)
            weight_sum = np.float32(0.0)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    sample = image[reflect101(y + dy, height), reflect101(x + dx, width)].astype(np.float32)
                    difference = sample - center
                    color_distance = np.sum(difference * difference, dtype=np.float32)
                    exponent = np.float32(
                        -(np.float32(dx * dx + dy * dy) * spatial_inv + color_distance * color_inv)
                    )
                    weight = np.float32(np.exp(exponent))
                    weight_sum = np.float32(weight_sum + weight)
                    sums = np.float32(sums + weight * sample)
            output[y, x] = np.floor(np.clip(sums / weight_sum, 0, 255) + np.float32(0.5)).astype(np.uint8)
    return output


exe = sys.argv[1]
rng = np.random.default_rng(20250308)
image = rng.integers(0, 256, (7, 9, 3), dtype=np.uint8)
radius, sigma_space, sigma_color = 2, 1.35, 31.0
expected = reference(image, radius, sigma_space, sigma_color)


def run(source_image, radius_value, sigma_space_value, sigma_color_value, root, name):
    source = root / (name + ".in")
    destination = root / (name + ".out")
    source.write_bytes(source_image.tobytes())
    subprocess.run(
        [exe, "bilateral_blur", str(source), str(destination), str(source_image.shape[1]),
         str(source_image.shape[0]), str(source_image.shape[2]), str(radius_value),
         str(sigma_space_value), str(sigma_color_value)],
        check=True,
    )
    return np.fromfile(destination, dtype=np.uint8).reshape(source_image.shape)


with tempfile.TemporaryDirectory() as temporary_directory:
    root = pathlib.Path(temporary_directory)
    np.testing.assert_array_equal(run(image, radius, sigma_space, sigma_color, root, "random"), expected)
    constant = np.full((3, 4, 2), 117, dtype=np.uint8)
    np.testing.assert_array_equal(run(constant, 3, 0.5, 2.0, root, "constant"), constant)
    singleton = np.array([[[249]]], dtype=np.uint8)
    np.testing.assert_array_equal(run(singleton, 1, 1.0, 1.0, root, "singleton"), singleton)
    got = run(image, 1, 1.0, 0.01, root, "edge_preserving")
    np.testing.assert_array_equal(got, image)
    assert np.all((got >= 0) & (got <= 255))
print("PASS: BilateralBlur direct-reference parity and properties")
