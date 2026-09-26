#!/usr/bin/env python3
"""Fixed-reference and property checks for the deterministic PlanckianJitter contract."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

exe = sys.argv[1]
image = np.array(
    [
        [[0, 64, 128, 7], [32, 96, 160, 8], [64, 128, 192, 9]],
        [[255, 200, 100, 10], [240, 120, 20, 11], [17, 34, 51, 12]],
    ],
    dtype=np.uint8,
)

def blackbody_rgb(kelvin):
    t = kelvin / 100.0
    if t <= 66.0:
        red = 255.0
        green = 99.4708025861 * np.log(t) - 161.1195681661
        blue = 0.0 if t <= 19.0 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * (t - 60.0) ** -0.1332047592
        green = 288.1221695283 * (t - 60.0) ** -0.0755148492
        blue = 255.0
    return np.clip([red, green, blue], 0.0, 255.0)

def reference(image, kelvin):
    gains = blackbody_rgb(kelvin) / blackbody_rgb(6500.0)
    rgb = np.floor(image[..., :3].astype(np.float64) * gains + 0.5)
    return np.concatenate((np.clip(rgb, 0, 255).astype(np.uint8), image[..., 3:]), axis=2)

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "in.raw"
    source.write_bytes(image.tobytes())

    warm = root / "warm.raw"
    subprocess.run(
        [exe, "color", "planckian_jitter", str(source), str(warm), "3", "2", "4", "4000"],
        check=True,
    )
    got = np.fromfile(warm, dtype=np.uint8).reshape(image.shape)
    np.testing.assert_allclose(got, reference(image, 4000.0), rtol=0, atol=1)
    np.testing.assert_array_equal(got[..., 3], image[..., 3])

    identity = root / "identity.raw"
    subprocess.run(
        [exe, "color", "planckian_jitter", str(source), str(identity), "3", "2", "4", "6500"],
        check=True,
    )
    np.testing.assert_array_equal(np.fromfile(identity, dtype=np.uint8), image.reshape(-1))

    for invalid in ("999", "40001", "nan"):
        result = subprocess.run(
            [exe, "color", "planckian_jitter", str(source), str(root / "bad.raw"), "3", "2", "4", invalid],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert result.returncode != 0, f"temperature {invalid} should be rejected"

# The normalized approximation is warm at 4000 K and cool at 10000 K.
assert blackbody_rgb(4000.0)[2] / blackbody_rgb(6500.0)[2] < 1.0
assert blackbody_rgb(10000.0)[2] / blackbody_rgb(6500.0)[2] >= 1.0
print("PASS: PlanckianJitter fixed-reference parity and properties")
