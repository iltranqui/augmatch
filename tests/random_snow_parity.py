#!/usr/bin/env python3
"""Fixed-reference and property checks for RandomSnow."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, snowflakes, alpha):
    result = bytearray(image)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                value = float(image[(y * width + x) * channels + channel])
                for sx, sy, radius, local_alpha in snowflakes:
                    dx, dy = x + 0.5 - sx, y + 0.5 - sy
                    if dx * dx + dy * dy <= radius * radius:
                        blend = max(0.0, min(1.0, alpha * local_alpha))
                        value = (1.0 - blend) * value + blend * 255.0
                result[(y * width + x) * channels + channel] = max(0, min(255, int(value + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 4, 3, 3
image = bytes((i * 29 + 7) % 256 for i in range(width * height * channels))
snowflakes = ((1.5, 1.5, 0.9, 0.5), (1.5, 1.5, 0.9, 1.0), (0.5, 0.5, 0.5, 0.25))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, records, result, expected = (root / name for name in ("input.raw", "flakes.bin", "output.raw", "expected.raw"))
    source.write_bytes(image)
    records.write_bytes(b"".join(struct.pack("<4f", *flake) for flake in snowflakes))
    expected.write_bytes(reference(image, width, height, channels, snowflakes, 0.5))
    subprocess.run([exe, "weather", "random_snow", str(source), str(result), str(width), str(height), str(channels), "3", ".5", "99", ".5", "2", str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomSnow reference mismatch")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_snow", str(source), str(first), str(width), str(height), str(channels), "32", ".7", "1234", ".5", "2.5"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded RandomSnow placement is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_snow", str(source), str(identity), str(width), str(height), str(channels), "3", "0", "0", ".5", "2"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomSnow alpha is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_snow", str(source), str(empty), str(width), str(height), str(channels), "0", ".5", "0"], check=True)
    if empty.read_bytes() != image:
        raise AssertionError("zero RandomSnow snowflake count is not identity")

print("PASS: RandomSnow fixed reference, deterministic seed, and alpha properties")
