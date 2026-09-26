#!/usr/bin/env python3
"""Compare deterministic Downscale's resize round trip with OpenCV."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np

exe = sys.argv[1]
rng = np.random.default_rng(20261008)
image = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
scale = 0.5
small = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
reference = cv2.resize(image, small, interpolation=cv2.INTER_LINEAR)
reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "input.raw"
    target = pathlib.Path(td) / "output.raw"
    image.tofile(source)
    subprocess.run(
        [exe, "downscale", str(source), str(target), "23", "17", "3", "0.5", "1", "1"],
        check=True,
    )
    actual = np.fromfile(target, dtype=np.uint8).reshape(image.shape)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1)
print("PASS: downscale resize round-trip parity within one uint8 level")
