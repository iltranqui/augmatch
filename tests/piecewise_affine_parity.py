#!/usr/bin/env python3
"""Parity for the deterministic PiecewiseAffine displacement-grid contract."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

exe = sys.argv[1]
h, w, channels = 8, 10, 3
image = (np.arange(h * w * channels, dtype=np.uint16) * 7 % 251).astype(np.uint8).reshape(h, w, channels)
dx = np.array([[0.0, 0.2, -0.1, 0.0], [0.1, 0.5, -0.3, 0.2], [0.0, -0.2, 0.4, 0.0]], dtype=np.float32)
dy = np.array([[0.0, -0.1, 0.2, 0.0], [0.2, 0.4, -0.2, 0.1], [0.0, 0.3, -0.1, 0.0]], dtype=np.float32)
gh, gw = dx.shape
xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
gx = xx * (gw - 1) / (w - 1)
gy = yy * (gh - 1) / (h - 1)
x0 = np.minimum(gw - 2, np.floor(gx).astype(np.int32))
y0 = np.minimum(gh - 2, np.floor(gy).astype(np.int32))
u, v = gx - x0, gy - y0
tl = y0 * gw + x0
tr, bl, br = tl + 1, tl + gw, tl + gw + 1
def piecewise(grid):
    flat = grid.ravel()
    return np.where(v <= u,
        (1-u)*flat[tl] + (u-v)*flat[tr] + v*flat[br],
        (1-v)*flat[tl] + u*flat[br] + (v-u)*flat[bl])
map_x, map_y = xx + piecewise(dx), yy + piecewise(dy)
reference = np.full(image.shape, 17, dtype=np.uint8)
for y in range(h):
    for x in range(w):
        sx, sy = float(map_x[y, x]), float(map_y[y, x])
        x0s, y0s = int(np.floor(sx)), int(np.floor(sy))
        ax, ay = sx - x0s, sy - y0s
        def at(px, py):
            return image[py, px].astype(np.float32) if 0 <= px < w and 0 <= py < h else np.full(channels, 17, dtype=np.float32)
        value = ((1-ay)*((1-ax)*at(x0s, y0s) + ax*at(x0s+1, y0s)) +
                 ay*((1-ax)*at(x0s, y0s+1) + ax*at(x0s+1, y0s+1)))
        reference[y, x] = np.clip(np.floor(value + 0.5), 0, 255).astype(np.uint8)
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source, output = root / "input.raw", root / "output.raw"
    xgrid, ygrid = root / "dx.f32", root / "dy.f32"
    image.tofile(source); dx.tofile(xgrid); dy.tofile(ygrid)
    subprocess.run([exe, "piecewise_affine", str(source), str(output), str(w), str(h),
                    str(channels), str(gw), str(gh), str(xgrid), str(ygrid), "17"], check=True)
    actual = np.fromfile(output, dtype=np.uint8).reshape(image.shape)
max_error = int(np.abs(actual.astype(np.int16) - reference.astype(np.int16)).max())
if max_error > 1:
    raise AssertionError(f"piecewise affine parity error {max_error} > 1")
print(f"PASS: PiecewiseAffine deterministic reference parity (max error {max_error})")
