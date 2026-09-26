#!/usr/bin/env python3
"""Exact channel-major LUT and deterministic-curve checks for RandomToneCurve."""
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
width, height, channels = 256, 1, 3
image = bytes(value for value in range(width) for _ in range(channels))
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source = root / "in.raw"
    source.write_bytes(image)

    lut = bytearray()
    for channel in range(channels):
        lut.extend((value if channel == 0 else 255 - value if channel == 1
                    else (value * 37 + 11) % 256) for value in range(256))
    lut_path = root / "lut.raw"
    lut_path.write_bytes(lut)
    explicit_out = root / "explicit.raw"
    subprocess.run(
        [exe, "tone", "random_tone_curve", str(source), str(explicit_out),
         str(width), str(height), str(channels), str(lut_path)], check=True)
    got = explicit_out.read_bytes()
    expected = bytearray()
    for pixel in range(width):
        for channel in range(channels):
            expected.append(lut[channel * 256 + pixel])
    assert got == bytes(expected), "explicit LUT output differs"

    identity_path = root / "identity.raw"
    identity = bytes(range(256)) * channels
    identity_path.write_bytes(identity)
    identity_out = root / "identity-out.raw"
    subprocess.run(
        [exe, "tone", "random_tone_curve", str(source), str(identity_out),
         str(width), str(height), str(channels), str(identity_path)], check=True)
    assert identity_out.read_bytes() == image, "identity LUT changed the image"

    generated_a = root / "generated-a.raw"
    generated_b = root / "generated-b.raw"
    command = [exe, "tone", "random_tone_curve", str(source), str(generated_a),
               str(width), str(height), str(channels), "-", "12345"]
    subprocess.run(command, check=True)
    command[4] = str(generated_b)
    subprocess.run(command, check=True)
    generated = generated_a.read_bytes()
    assert generated == generated_b.read_bytes(), "seeded curves are not repeatable"
    for channel in range(channels):
        curve = [generated[value * channels + channel] for value in range(256)]
        assert curve[0] == 0 and curve[-1] == 255
        assert all(a <= b for a, b in zip(curve, curve[1:]))

print("PASS: RandomToneCurve exact channel-major LUT and deterministic curve properties")
