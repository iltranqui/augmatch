#!/usr/bin/env python3
"""Fixed-reference and property checks for the deterministic FancyPCA contract."""
import math
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
width, height, channels = 3, 2, 4
image = bytes([
    0, 64, 128, 7, 32, 96, 160, 8, 64, 128, 192, 9,
    255, 200, 100, 10, 240, 120, 20, 11, 17, 34, 51, 12,
])
basis = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
eigenvalues = [10.0, 20.0, 30.0]
perturbation = [0.5, -0.25, 1.0]
alpha = 0.5

def run(root, output, alpha_arg=None):
    args = [exe, "color", "fancy_pca", str(root / "in.raw"), str(output),
            str(width), str(height), str(channels)]
    args += [str(value) for value in basis + eigenvalues + perturbation]
    if alpha_arg is not None:
        args.append(str(alpha_arg))
    subprocess.run(args, check=True)

def reference(alpha_value):
    delta = []
    for row in range(3):
        delta.append(sum(basis[row * 3 + col] * eigenvalues[col] *
                         perturbation[col] * alpha_value for col in range(3)))
    output = bytearray(image)
    for i in range(width * height):
        for channel in range(3):
            output[i * channels + channel] = max(
                0, min(255, int(math.floor(image[i * channels + channel] +
                                            delta[channel] + 0.5))))
    return bytes(output)

with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    (root / "in.raw").write_bytes(image)
    output = root / "out.raw"
    run(root, output, alpha)
    assert output.read_bytes() == reference(alpha), "FancyPCA fixed-reference mismatch"
    assert output.read_bytes()[3::4] == image[3::4], "non-RGB channels must be copied"

    default_alpha = root / "default_alpha.raw"
    run(root, default_alpha)
    assert default_alpha.read_bytes() == reference(1.0), "omitted alpha must default to one"

    identity = root / "identity.raw"
    run(root, identity, 0.0)
    assert identity.read_bytes() == image, "alpha=0 must be identity"

    repeat = root / "repeat.raw"
    run(root, repeat, alpha)
    assert repeat.read_bytes() == output.read_bytes(), "explicit parameters must be repeatable"

    bad = subprocess.run(
        [exe, "color", "fancy_pca", str(root / "in.raw"), str(root / "bad.raw"),
         str(width), str(height), str(channels)] + ["0"] * 15,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert bad.returncode != 0, "a non-orthonormal basis must be rejected"

print("PASS: FancyPCA fixed-reference parity and properties")
