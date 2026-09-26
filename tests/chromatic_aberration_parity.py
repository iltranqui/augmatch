#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np

exe = sys.argv[1]
rng = np.random.default_rng(173)
h, w, channels = 13, 17, 4
image = rng.integers(0, 256, (h, w, channels), dtype=np.uint8)
radial = (0.075, -0.041, 0.019)
gain = (1.08, 0.91, 1.03)
fill = 7
cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
fx, fy = max(1.0, w * 0.5), max(1.0, h * 0.5)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
xn, yn = (xx - cx) / fx, (yy - cy) / fy
r2 = xn * xn + yn * yn
reference = np.empty_like(image)
for channel in range(3):
    scale = 1.0 + radial[channel] * r2
    map_x = (cx + xn * scale * fx).astype(np.float32)
    map_y = (cy + yn * scale * fy).astype(np.float32)
    sampled = cv2.remap(image[:, :, channel], map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    reference[:, :, channel] = np.clip(np.floor(sampled.astype(np.float32) * gain[channel] + 0.5), 0, 255)
reference[:, :, 3] = image[:, :, 3]

with tempfile.TemporaryDirectory() as td:
    src = pathlib.Path(td) / "in.raw"
    dst = pathlib.Path(td) / "out.raw"
    image.tofile(src)
    subprocess.run([
        exe, "color", "chromatic_aberration", str(src), str(dst), str(w), str(h), str(channels),
        *(str(value) for value in (*radial, *gain, fill)),
    ], check=True)
    got = np.fromfile(dst, dtype=np.uint8).reshape(image.shape)

np.testing.assert_allclose(got, reference, atol=2, rtol=0)
print("PASS: chromatic aberration OpenCV remap parity")
