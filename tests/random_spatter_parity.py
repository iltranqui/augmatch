#!/usr/bin/env python3
"""Fixed-reference and property checks for Spatter."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, droplets, alpha):
    result = bytearray(image)
    for y in range(height):
        for x in range(width):
            for channel in range(min(channels, 3)):
                value = float(image[(y * width + x) * channels + channel])
                for dx0, dy0, radius, red, green, blue, local_alpha in droplets:
                    dx, dy = x + 0.5 - dx0, y + 0.5 - dy0
                    if dx * dx + dy * dy <= radius * radius:
                        color = (red, green, blue)[channel]
                        blend = max(0.0, min(1.0, alpha * local_alpha))
                        value = (1.0 - blend) * value + blend * color
                result[(y * width + x) * channels + channel] = max(0, min(255, int(value + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 5, 4, 4
image = bytes((i * 31 + 9) % 256 for i in range(width * height * channels))
droplets = ((1.5, 1.5, 0.9, 230, 20, 40, 0.5),
            (1.5, 1.5, 0.9, 10, 210, 60, 1.0),
            (3.5, 2.5, 1.1, 40, 50, 220, 0.75))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, records, result, expected = (root / name for name in
                                          ("input.raw", "droplets.bin", "output.raw", "expected.raw"))
    source.write_bytes(image)
    # SpatterDroplet is three float32 values, three color bytes, one ABI pad byte,
    # and one float32 alpha (sizeof(SpatterDroplet) is 20 on supported targets).
    records.write_bytes(b"".join(struct.pack("<3f3Bxf", *droplet) for droplet in droplets))
    expected.write_bytes(reference(image, width, height, channels, droplets, 0.5))
    subprocess.run([exe, "weather", "spatter", str(source), str(result), str(width), str(height), str(channels),
                    "3", ".5", "99", ".5", "2", "255", "255", "255", str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit Spatter reference mismatch")
    if any(result.read_bytes()[i] != image[i] for i in range(3, len(image), 4)):
        raise AssertionError("Spatter changed channels beyond RGB")

    first, second = root / "seed-a.raw", root / "seed-b.raw"
    command = [exe, "random_spatter", str(source), str(first), str(width), str(height), str(channels),
               "32", ".7", "1234", ".5", "2.5", "220", "30", "10"]
    subprocess.run(command, check=True)
    command[3] = str(second)
    subprocess.run(command, check=True)
    if first.read_bytes() != second.read_bytes():
        raise AssertionError("seeded Spatter placement is not deterministic")

    identity = root / "identity.raw"
    subprocess.run([exe, "spatter", str(source), str(identity), str(width), str(height), str(channels),
                    "3", "0", "0", ".5", "2", "1", "2", "3"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero Spatter alpha is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_spatter", str(source), str(empty), str(width), str(height), str(channels),
                    "0", ".5", "0"], check=True)
    if empty.read_bytes() != image:
        raise AssertionError("zero Spatter droplet count is not identity")

print("PASS: Spatter fixed reference, deterministic seed, RGB color, and alpha properties")
