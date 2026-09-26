#!/usr/bin/env python3
"""Parity for the explicit displacement-grid contract, using OpenCV remap."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np

exe = sys.argv[1]
h, w, channels = 7, 9, 3
image = np.arange(h * w * channels, dtype=np.uint8).reshape(h, w, channels)
# 3x3 vertex displacements in source-pixel units, row-major (y, x).
dx = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0],
], dtype=np.float32)
dy = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.25, 0.0],
    [0.0, 0.0, 0.0],
], dtype=np.float32)
xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
def interpolate_grid(grid):
    gx = xx * (grid.shape[1] - 1) / (w - 1)
    gy = yy * (grid.shape[0] - 1) / (h - 1)
    x0 = np.minimum(grid.shape[1] - 2, np.floor(gx).astype(np.int32))
    y0 = np.minimum(grid.shape[0] - 2, np.floor(gy).astype(np.int32))
    ax, ay = gx - x0, gy - y0
    return ((1 - ay) * ((1 - ax) * grid[y0, x0] + ax * grid[y0, x0 + 1]) +
            ay * ((1 - ax) * grid[y0 + 1, x0] + ax * grid[y0 + 1, x0 + 1]))
# The native operation bilinearly interpolates vertex displacement, then uses
# OpenCV's source-map remapper at (x + dx, y + dy).
map_x = (xx + interpolate_grid(dx)).astype(np.float32)
map_y = (yy + interpolate_grid(dy)).astype(np.float32)
reference = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=17)
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "input.raw"
    output = root / "output.raw"
    xgrid = root / "dx.f32"
    ygrid = root / "dy.f32"
    image.tofile(source)
    dx.tofile(xgrid)
    dy.tofile(ygrid)
    subprocess.run([exe, "grid_distortion", str(source), str(output),
                    str(w), str(h), str(channels), "3", "3",
                    str(xgrid), str(ygrid), "17"], check=True)
    actual = np.fromfile(output, dtype=np.uint8).reshape(image.shape)

max_error = int(np.abs(actual.astype(np.int16) - reference.astype(np.int16)).max())
if max_error > 1:
    raise AssertionError(f"grid distortion parity error {max_error} > 1")
print(f"PASS: GridDistortion OpenCV remap parity (max error {max_error})")
