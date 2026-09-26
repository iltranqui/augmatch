#!/usr/bin/env python3
"""Compare the native resize primitive with Albumentations/OpenCV."""
import pathlib
import subprocess
import sys
import tempfile
import numpy as np
import albumentations as A

exe = sys.argv[1]
rng = np.random.default_rng(20260924)
image = rng.integers(0, 256, (17, 23, 3), dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "input.raw"
    target = pathlib.Path(td) / "output.raw"
    image.tofile(source)
    reference = A.Resize(height=11, width=13, interpolation=1, p=1.0)(image=image)["image"]
    subprocess.run([exe, "resize", str(source), str(target), "23", "17", "3", "13", "11", "1"], check=True)
    actual = np.fromfile(target, dtype=np.uint8).reshape(reference.shape)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1)
print("PASS: resize parity within one uint8 level")
