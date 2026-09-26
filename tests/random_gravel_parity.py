#!/usr/bin/env python3
"""Fixed-reference and property checks for RandomGravel."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, particles, alpha):
    result = bytearray(image)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                value = float(image[(y * width + x) * channels + channel])
                for px, py, radius, local_alpha, fill in particles:
                    dx, dy = x + 0.5 - px, y + 0.5 - py
                    if dx * dx + dy * dy <= radius * radius:
                        blend = max(0.0, min(1.0, alpha * local_alpha))
                        value = (1.0 - blend) * value + blend * fill
                result[(y * width + x) * channels + channel] = max(0, min(255, int(value + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 5, 4, 3
image = bytes((i * 31 + 9) % 256 for i in range(width * height * channels))
particles = ((1.5, 1.5, 0.9, 0.5, 32), (1.5, 1.5, 0.9, 1.0, 220),
             (3.5, 2.5, 1.1, 0.75, 96))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, records, result, expected = (root / name for name in
                                          ("input.raw", "particles.bin", "output.raw", "expected.raw"))
    source.write_bytes(image)
    # GravelParticle has four float32 fields followed by uint8 fill and three
    # ABI padding bytes (sizeof(GravelParticle) is 20 on supported targets).
    records.write_bytes(b"".join(struct.pack("<4fB3x", *particle) for particle in particles))
    expected.write_bytes(reference(image, width, height, channels, particles, 0.5))
    subprocess.run([exe, "weather", "random_gravel", str(source), str(result),
                    str(width), str(height), str(channels), "3", ".5", "99",
                    ".5", "2", "128", str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomGravel reference mismatch")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_gravel", str(source), str(first), str(width), str(height),
               str(channels), "32", ".7", "1234", ".5", "2.5", "77"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded RandomGravel placement is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_gravel", str(source), str(identity), str(width), str(height),
                    str(channels), "3", "0", "0", ".5", "2", "128"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomGravel alpha is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_gravel", str(source), str(empty), str(width), str(height),
                    str(channels), "0", ".5", "0"], check=True)
    if empty.read_bytes() != image:
        raise AssertionError("zero RandomGravel particle count is not identity")

print("PASS: RandomGravel fixed reference, deterministic seed, and alpha properties")
