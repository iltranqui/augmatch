#!/usr/bin/env python3
"""Fixed-reference, deterministic-field, and veil properties for RandomFog."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, field, density, opacity):
    result = bytearray(image)
    for pixel, local_density in enumerate(field):
        blend = max(0.0, min(1.0, density * opacity * local_density))
        for channel in range(channels):
            value = image[pixel * channels + channel]
            result[pixel * channels + channel] = max(0, min(255, int((1.0 - blend) * value + blend * 255.0 + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 3, 2, 3
image = bytes((i * 37 + 11) % 256 for i in range(width * height * channels))
field = (0.0, 0.25, 0.5, 0.75, 1.0, 0.4)
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, field_file, result, expected = (root / name for name in ("input.raw", "field.f32", "output.raw", "expected.raw"))
    source.write_bytes(image)
    field_file.write_bytes(b"".join(struct.pack("<f", value) for value in field))
    expected.write_bytes(reference(image, width, height, channels, field, 0.8, 0.5))
    subprocess.run([exe, "weather", "random_fog", str(source), str(result), str(width), str(height), str(channels), ".8", ".5", "99", str(field_file)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomFog field reference mismatch")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_fog", str(source), str(first), str(width), str(height), str(channels), ".7", ".6", "1234"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded RandomFog field is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_fog", str(source), str(identity), str(width), str(height), str(channels), ".7", "0", "0"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomFog opacity is not identity")

    zero_density = root / "zero-density.raw"
    subprocess.run([exe, "random_fog", str(source), str(zero_density), str(width), str(height), str(channels), "0", ".8", "0"], check=True)
    if zero_density.read_bytes() != image:
        raise AssertionError("zero RandomFog density is not identity")

print("PASS: RandomFog fixed reference, deterministic seed, and veil properties")
