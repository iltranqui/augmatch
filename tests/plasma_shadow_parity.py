#!/usr/bin/env python3
"""Reference and property checks for the deterministic PlasmaShadow contract."""
import math
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, field, threshold, strength, channels):
    out = bytearray()
    for index, value in enumerate(image):
        plasma = field[index // channels]
        shadow = ((threshold - plasma) / threshold) if threshold > 0.0 and plasma < threshold else 0.0
        factor = 1.0 - strength * shadow
        out.append(max(0, min(255, int(math.floor(value * factor + 0.5)))))
    return bytes(out)


exe = sys.argv[1]
w, h, channels = 5, 2, 3
image = bytes((index * 41 + 17) % 256 for index in range(w * h * channels))
field = (0.0, 0.2, 0.5, 0.8, 1.0, 0.125, 0.875, 0.375, 0.625, 0.25)
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "input.raw"
    explicit = root / "field.f32"
    source.write_bytes(image)
    explicit.write_bytes(b"".join(struct.pack("<f", value) for value in field))

    result = root / "explicit.raw"
    expected = root / "expected.raw"
    expected.write_bytes(reference(image, field, 0.75, 0.8, channels))
    subprocess.run([exe, "color", "plasma_shadow", str(source), str(result),
                    str(w), str(h), str(channels), "0.75", "0.8", "999",
                    str(explicit)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit plasma shadow reference mismatch")

    seeded_a, seeded_b = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "color", "plasma_shadow", str(source), str(seeded_a),
               str(w), str(h), str(channels), "0.6", "0.7", "12345"]
    subprocess.run(command, check=True)
    command[4] = str(seeded_b)
    subprocess.run(command, check=True)
    if seeded_a.read_bytes() != seeded_b.read_bytes():
        raise AssertionError("fixed plasma shadow seed is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "color", "plasma_shadow", str(source), str(identity),
                    str(w), str(h), str(channels), "0", "1", "0"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero threshold does not disable plasma shadow")

    identity = root / "zero-strength.raw"
    subprocess.run([exe, "color", "plasma_shadow", str(source), str(identity),
                    str(w), str(h), str(channels), "0.5", "0", "0"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero shadow strength is not identity")

print("PASS: PlasmaShadow reference, fixed-seed, and property checks")
