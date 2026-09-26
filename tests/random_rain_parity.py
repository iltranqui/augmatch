#!/usr/bin/env python3
"""Reference, deterministic-placement, and alpha properties for RandomRain."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, streaks, alpha):
    result = bytearray(image)
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                value = float(image[(y * width + x) * channels + channel])
                for x0, y0, x1, y1, width_streak, local_alpha in streaks:
                    dx, dy = x1 - x0, y1 - y0
                    length2 = dx * dx + dy * dy
                    t = 0.0 if length2 == 0.0 else max(0.0, min(1.0, ((x + .5 - x0) * dx + (y + .5 - y0) * dy) / length2))
                    ex, ey = x + .5 - (x0 + t * dx), y + .5 - (y0 + t * dy)
                    if ex * ex + ey * ey <= .25 * width_streak * width_streak:
                        blend = max(0.0, min(1.0, alpha * local_alpha))
                        value = (1.0 - blend) * value + blend * 255.0
                result[(y * width + x) * channels + channel] = max(0, min(255, int(value + .5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 4, 3, 3
image = bytes((i * 29 + 7) % 256 for i in range(width * height * channels))
streaks = ((1.5, -1.0, 1.5, 4.0, 1.0, 0.5), (0.0, 0.0, 3.0, 2.0, 1.2, 1.0))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, records, result, expected = (root / name for name in ("input.raw", "streaks.bin", "output.raw", "expected.raw"))
    source.write_bytes(image)
    records.write_bytes(b"".join(struct.pack("<6f", *streak) for streak in streaks))
    expected.write_bytes(reference(image, width, height, channels, streaks, .5))
    subprocess.run([exe, "weather", "random_rain", str(source), str(result), str(width), str(height), str(channels), "2", ".5", "99", "4", "8", "75", "105", "1", str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomRain reference mismatch")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_rain", str(source), str(first), str(width), str(height), str(channels), "12", ".7", "1234", "2", "5", "80", "100", "1"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded RandomRain placement is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_rain", str(source), str(identity), str(width), str(height), str(channels), "2", "0", "0", "4", "8", "75", "105", "1"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomRain alpha is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_rain", str(source), str(empty), str(width), str(height), str(channels), "0", ".5", "0"], check=True)
    if empty.read_bytes() != image:
        raise AssertionError("zero RandomRain streak count is not identity")

print("PASS: RandomRain fixed reference, deterministic seed, and alpha properties")
