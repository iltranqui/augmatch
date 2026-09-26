#!/usr/bin/env python3
"""Fixed-reference and property checks for the deterministic RandomShadow contract."""
import math
import pathlib
import struct
import subprocess
import sys
import tempfile


def inside_polygon(px, py, points):
    inside = False
    for i, b in enumerate(points):
        a = points[i - 1]
        cross = (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])
        if abs(cross) <= 1e-6 and min(a[0], b[0]) <= px <= max(a[0], b[0]) and min(a[1], b[1]) <= py <= max(a[1], b[1]):
            return True
        if (a[1] > py) != (b[1] > py) and px < (b[0] - a[0]) * (py - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def reference(image, width, height, channels, polygons, rectangles, opacity, fill):
    result = bytearray()
    for y in range(height):
        for x in range(width):
            px, py = x + 0.5, y + 0.5
            for channel in range(channels):
                value = float(image[(y * width + x) * channels + channel])
                for points, alpha in polygons:
                    if inside_polygon(px, py, points):
                        a = max(0.0, min(1.0, opacity * alpha))
                        value = (1.0 - a) * value + a * fill
                for x0, y0, x1, y1, alpha in rectangles:
                    if min(x0, x1) <= px <= max(x0, x1) and min(y0, y1) <= py <= max(y0, y1):
                        a = max(0.0, min(1.0, opacity * alpha))
                        value = (1.0 - a) * value + a * fill
                result.append(max(0, min(255, int(math.floor(value + 0.5)))))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 7, 5, 3
image = bytes((i * 29 + 17) % 256 for i in range(width * height * channels))
polygons = [([(1.0, 0.5), (5.5, 1.0), (3.0, 4.5)], 0.75),
            ([(0.5, 3.5), (2.5, 3.5), (2.5, 4.5), (0.5, 4.5)], 0.25)]
rectangles = [(4.5, 0.5, 6.5, 2.5, 0.5), (1.5, 1.5, 3.5, 3.5, 1.0)]
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, polygon_file, rectangle_file, result, expected = (root / name for name in ("input.raw", "polygons.bin", "rectangles.bin", "output.raw", "expected.raw"))
    source.write_bytes(image)
    with polygon_file.open("wb") as stream:
        for points, alpha in polygons:
            stream.write(struct.pack("<If", len(points), alpha))
            stream.write(b"".join(struct.pack("<2f", *point) for point in points))
    rectangle_file.write_bytes(b"".join(struct.pack("<5f", *rectangle) for rectangle in rectangles))
    expected.write_bytes(reference(image, width, height, channels, polygons, rectangles, 0.8, 32))
    subprocess.run([exe, "weather", "random_shadow", str(source), str(result), str(width), str(height), str(channels), ".8", "32", str(polygon_file), str(rectangle_file)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("explicit RandomShadow reference mismatch")

    identity = root / "identity.raw"
    subprocess.run([exe, "random_shadow", str(source), str(identity), str(width), str(height), str(channels), "0", "0", str(polygon_file), str(rectangle_file)], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero RandomShadow opacity is not identity")

    empty = root / "empty.raw"
    subprocess.run([exe, "random_shadow", str(source), str(empty), str(width), str(height), str(channels), ".5", "0", "-", "-"], check=True)
    if empty.read_bytes() != image:
        raise AssertionError("empty RandomShadow masks are not identity")

print("PASS: RandomShadow fixed reference and opacity/mask properties")
