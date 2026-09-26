#!/usr/bin/env python3
"""Reference and property checks for the deterministic PlasmaBrightnessContrast contract."""
import math
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, field, brightness, contrast, strength, channels):
    out = bytearray()
    for index, value in enumerate(image):
        plasma = field[index // channels]
        factor = contrast * (1.0 + strength * (2.0 * plasma - 1.0))
        result = 127.5 + (value - 127.5) * factor + brightness * 255.0
        out.append(max(0, min(255, int(math.floor(result + 0.5)))))
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
    expected.write_bytes(reference(image, field, 0.125, 1.25, 0.8, channels))
    subprocess.run([exe, "color", "plasma_brightness_contrast", str(source), str(result),
                    str(w), str(h), str(channels), "0.125", "1.25", "0.8", "999",
                    str(explicit)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit plasma brightness/contrast reference mismatch")

    seeded_a, seeded_b = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "color", "plasma_brightness_contrast", str(source), str(seeded_a),
               str(w), str(h), str(channels), "-0.07", "1.35", "0.7", "12345"]
    subprocess.run(command, check=True)
    command[4] = str(seeded_b)
    subprocess.run(command, check=True)
    if seeded_a.read_bytes() != seeded_b.read_bytes():
        raise AssertionError("fixed plasma seed is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "color", "plasma_brightness_contrast", str(source), str(identity),
                    str(w), str(h), str(channels), "0", "1", "0", "0"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero brightness and plasma strength is not identity contrast")

    contrast = root / "contrast.raw"
    subprocess.run([exe, "color", "plasma_contrast", str(source), str(contrast),
                    str(w), str(h), str(channels), "1.35", "0.7", "12345"], check=True)
    zero_brightness = root / "zero-brightness.raw"
    subprocess.run([exe, "color", "plasma_brightness_contrast", str(source), str(zero_brightness),
                    str(w), str(h), str(channels), "0", "1.35", "0.7", "12345"], check=True)
    if zero_brightness.read_bytes() != contrast.read_bytes():
        raise AssertionError("zero brightness does not reuse PlasmaContrast semantics")

print("PASS: PlasmaBrightnessContrast reference, fixed-seed, and property checks")
