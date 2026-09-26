#!/usr/bin/env python3
"""Reference and property checks for the explicit PlasmaContrast contract."""
import pathlib
import struct
import subprocess
import sys
import tempfile
import math


def reference(image, field, contrast, strength):
    out = bytearray()
    for value, noise in zip(image, field):
        factor = contrast * (1.0 + strength * (2.0 * noise - 1.0))
        out.append(max(0, min(255, int(math.floor(127.5 + (value - 127.5) * factor + 0.5)))))
    return bytes(out)


exe = sys.argv[1]
w, h, channels = 4, 2, 3
image = bytes((i * 37 + 11) % 256 for i in range(w * h * channels))
field = (0.0, 0.25, 0.5, 0.75, 1.0, 0.125, 0.875, 0.375)
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "input.raw"
    explicit = root / "field.f32"
    expected = root / "expected.raw"
    result = root / "result.raw"
    source.write_bytes(image)
    explicit.write_bytes(b"".join(struct.pack("<f", value) for value in field))
    expected.write_bytes(reference(image, [field[i // channels] for i in range(len(image))], 1.25, 0.8))
    subprocess.run([exe, "color", "plasma_contrast", str(source), str(result), str(w), str(h), str(channels), "1.25", "0.8", "999", str(explicit)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit plasma field reference mismatch")
    seeded_a = root / "seed-a.raw"
    seeded_b = root / "seed-b.raw"
    subprocess.run([exe, "color", "plasma_contrast", str(source), str(seeded_a), str(w), str(h), str(channels), "1.35", "0.7", "12345"], check=True)
    subprocess.run([exe, "color", "plasma_contrast", str(source), str(seeded_b), str(w), str(h), str(channels), "1.35", "0.7", "12345"], check=True)
    if seeded_a.read_bytes() != seeded_b.read_bytes():
        raise AssertionError("fixed seed is not deterministic")
    identity = root / "identity.raw"
    subprocess.run([exe, "color", "plasma_contrast", str(source), str(identity), str(w), str(h), str(channels), "1", "0", "0"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero plasma strength is not identity contrast")
print("PASS: PlasmaContrast reference, fixed-seed, and property checks")
