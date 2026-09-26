#!/usr/bin/env python3
"""Parity for deterministic ColorJitter against a fixed OpenCV RGB/HSV pipeline."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np

exe = sys.argv[1]
rng = np.random.default_rng(123)
image = rng.integers(0, 256, (13, 17, 4), dtype=np.uint8)
brightness, contrast, saturation, hue = 0.125, 1.18, 0.72, 13.0

# The native contract uses RGB input, normalized additive brightness, multiplicative
# contrast/saturation, and OpenCV's 0..179 half-degree hue representation.
rgb = np.floor(image[..., :3].astype(np.float32) * contrast + brightness * 255.0 + 0.5)
rgb = np.clip(rgb, 0, 255).astype(np.uint8)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
hsv[..., 0] = np.mod(np.floor(hsv[..., 0] + hue + 0.5), 180.0)
hsv[..., 1] = np.clip(np.floor(hsv[..., 1] * saturation + 0.5), 0, 255)
reference = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
expected = np.concatenate((reference, image[..., 3:]), axis=2)

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "in.raw"
    output = root / "out.raw"
    source.write_bytes(image.tobytes())
    subprocess.run(
        [exe, "color", "color_jitter", str(source), str(output), "17", "13", "4",
         str(brightness), str(contrast), str(saturation), str(hue)],
        check=True,
    )
    got = np.fromfile(output, dtype=np.uint8).reshape(image.shape)

np.testing.assert_allclose(got, expected, rtol=0, atol=5)
assert np.array_equal(got[..., 3], image[..., 3]), "extra channels must be copied"
print("PASS: ColorJitter parity within five uint8 levels")
