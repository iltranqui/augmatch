#!/usr/bin/env python3
"""Pinned AdvancedBlur parity against an explicit NumPy/OpenCV reference."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def gaussian_kernel(size, sigma_x, sigma_y, angle_degrees):
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    angle = np.float32(np.deg2rad(angle_degrees))
    co, si = np.cos(angle), np.sin(angle)
    xp = co * xx + si * yy
    yp = -si * xx + co * yy
    kernel = np.exp(
        np.float32(-0.5)
        * (xp * xp / np.float32(sigma_x * sigma_x) + yp * yp / np.float32(sigma_y * sigma_y))
    ).astype(np.float32)
    return kernel / np.sum(kernel, dtype=np.float32)


exe = sys.argv[1]
rng = np.random.default_rng(20250308)
image = rng.integers(0, 256, (11, 13, 3), dtype=np.uint8)
size, sigma_x, sigma_y, angle = 7, 1.25, 2.0, 23.0
kernel = gaussian_kernel(size, sigma_x, sigma_y, angle)
reference_f32 = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
# C++ std::lround rounds nonnegative half values away from zero.
reference = np.clip(np.floor(reference_f32 + np.float32(0.5)), 0, 255).astype(np.uint8)

with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "in.raw"
    destination = pathlib.Path(td) / "out.raw"
    image.tofile(source)
    subprocess.run(
        [
            exe,
            "advanced_blur",
            str(source),
            str(destination),
            "13",
            "11",
            "3",
            str(size),
            str(sigma_x),
            str(sigma_y),
            str(angle),
        ],
        check=True,
    )
    got = np.fromfile(destination, dtype=np.uint8).reshape(image.shape)

np.testing.assert_array_equal(got, reference)
print("PASS: AdvancedBlur pinned NumPy/OpenCV parity")
