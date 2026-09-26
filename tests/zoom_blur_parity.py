#!/usr/bin/env python3
"""Deterministic ZoomBlur parity against a NumPy/OpenCV-defined reference."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def reflect101(index, size):
    # Delegate the border-index contract to OpenCV's named implementation.
    return cv2.borderInterpolate(index, size, cv2.BORDER_REFLECT_101)


def reference(image, minimum, maximum, steps):
    height, width, channels = image.shape
    # OpenCV's documented BORDER_REFLECT_101 convention is used for samples.
    assert cv2.BORDER_REFLECT_101 == 4
    center_x = np.float32(0.5 * (width - 1))
    center_y = np.float32(0.5 * (height - 1))
    result = np.empty_like(image)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                total = np.float32(0.0)
                for step in range(steps + 1):
                    fraction = np.float32(step / steps) if steps else np.float32(0.0)
                    factor = np.float32(minimum) + np.float32(maximum - minimum) * fraction
                    source_x = center_x + (np.float32(x) - center_x) / factor
                    source_y = center_y + (np.float32(y) - center_y) / factor
                    x0, y0 = int(np.floor(source_x)), int(np.floor(source_y))
                    fx, fy = source_x - x0, source_y - y0
                    xa, xb = reflect101(x0, width), reflect101(x0 + 1, width)
                    ya, yb = reflect101(y0, height), reflect101(y0 + 1, height)
                    top_left = np.float32(image[ya, xa, channel])
                    top_right = np.float32(image[ya, xb, channel])
                    bottom_left = np.float32(image[yb, xa, channel])
                    bottom_right = np.float32(image[yb, xb, channel])
                    top = top_left + (top_right - top_left) * fx
                    bottom = bottom_left + (bottom_right - bottom_left) * fx
                    total += top + (bottom - top) * fy
                result[y, x, channel] = np.uint8(np.floor(total / np.float32(steps + 1) + np.float32(0.5)))
    return result


exe = sys.argv[1]
rng = np.random.default_rng(20250308)
# cv2.merge makes the HWC test fixture through the same channel convention used by CLI data.
image = cv2.merge([rng.integers(0, 256, (9, 11), dtype=np.uint8) for _ in range(3)])
minimum, maximum, steps = 1.0, 1.30, 6
expected = reference(image, minimum, maximum, steps)
with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "in.raw"
    destination = pathlib.Path(td) / "out.raw"
    image.tofile(source)
    subprocess.run(
        [exe, "zoom_blur", str(source), str(destination), "11", "9", "3",
         str(minimum), str(maximum), str(steps)],
        check=True,
    )
    got = np.fromfile(destination, dtype=np.uint8).reshape(image.shape)
np.testing.assert_array_equal(got, expected)
print("PASS: ZoomBlur deterministic NumPy/OpenCV reference parity")
