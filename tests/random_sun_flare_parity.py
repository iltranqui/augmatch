#!/usr/bin/env python3
"""Fixed-reference and property checks for RandomSunFlare."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, source, rays, opacity):
    result = bytearray(image)
    sx, sy, radius, source_alpha = source
    for y in range(height):
        for x in range(width):
            px, py = x + 0.5, y + 0.5
            for channel in range(channels):
                value = float(image[(y * width + x) * channels + channel])
                if (px - sx) ** 2 + (py - sy) ** 2 <= radius ** 2:
                    a = max(0.0, min(1.0, opacity * source_alpha))
                    value = (1.0 - a) * value + a * 255.0
                for angle, length, width_, local_alpha in rays:
                    import math
                    theta = math.radians(angle)
                    dx, dy = length * math.cos(theta), length * math.sin(theta)
                    segment_length = dx * dx + dy * dy
                    t = 0.0 if segment_length == 0 else max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / segment_length))
                    ex, ey = px - (sx + t * dx), py - (sy + t * dy)
                    if ex * ex + ey * ey <= 0.25 * width_ * width_:
                        a = max(0.0, min(1.0, opacity * local_alpha))
                        value = (1.0 - a) * value + a * 255.0
                result[(y * width + x) * channels + channel] = max(0, min(255, int(value + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 7, 5, 3
image = bytes((i * 31 + 13) % 256 for i in range(width * height * channels))
source = (3.5, 2.5, 1.1, 0.6)
rays = ((0.0, 3.0, 1.1, 0.5), (90.0, 3.0, 1.5, 1.0), (180.0, 2.0, 0.8, 0.25))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source_file, records, result, expected = (root / name for name in ("input.raw", "rays.bin", "output.raw", "expected.raw"))
    source_file.write_bytes(image)
    records.write_bytes(b"".join(struct.pack("<4f", *ray) for ray in rays))
    expected.write_bytes(reference(image, width, height, channels, source, rays, 0.8))
    subprocess.run([exe, "weather", "random_sun_flare", str(source_file), str(result), str(width), str(height), str(channels), "3", ".8", "99", "3.5", "2.5", "1.1", ".6", "2", "4", "0", "360", "1", "2", str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomSunFlare reference mismatch")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_sun_flare", str(source_file), str(first), str(width), str(height), str(channels), "32", ".7", "1234", "3.5", "2.5", "0", "1", "1", "5", "0", "360", "1", "2"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded RandomSunFlare placement is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_sun_flare", str(source_file), str(identity), str(width), str(height), str(channels), "3", "0", "0", "3.5", "2.5", "1.1", "1"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomSunFlare opacity is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_sun_flare", str(source_file), str(empty), str(width), str(height), str(channels), "0", ".5", "0", "3.5", "2.5", "1", "1"], check=True)
    expected_source = reference(image, width, height, channels, (3.5, 2.5, 1, 1), (), .5)
    if empty.read_bytes() != expected_source:
        raise AssertionError("zero-ray RandomSunFlare source contract mismatch")

print("PASS: RandomSunFlare fixed reference, deterministic seed, and opacity properties")
