#!/usr/bin/env python3
"""Pinned parity for the deterministic RingingOvershoot high-boost filter."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def gaussian_kernel(size, sigma):
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    weights = np.exp(
        -(xx * xx + yy * yy).astype(np.float32) / np.float32(2.0 * sigma * sigma)
    ).astype(np.float32)
    return weights / np.sum(weights, dtype=np.float32)


exe = sys.argv[1]
rng = np.random.default_rng(20250308)
image = rng.integers(0, 256, (13, 11, 3), dtype=np.uint8)
size, sigma, amount = 5, 1.1, 1.25
kernel = gaussian_kernel(size, sigma)
blur = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
reference = np.clip(
    np.floor(image.astype(np.float32) + np.float32(amount) * (image.astype(np.float32) - blur) + np.float32(0.5)),
    0,
    255,
).astype(np.uint8)

with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "in.raw"
    destination = pathlib.Path(td) / "out.raw"
    image.tofile(source)
    subprocess.run(
        [
            exe,
            "ringing_overshoot",
            str(source),
            str(destination),
            "11",
            "13",
            "3",
            str(size),
            str(sigma),
            str(amount),
        ],
        check=True,
    )
    got = np.fromfile(destination, dtype=np.uint8).reshape(image.shape)

np.testing.assert_array_equal(got, reference)
print("PASS: RingingOvershoot pinned NumPy/OpenCV parity")
