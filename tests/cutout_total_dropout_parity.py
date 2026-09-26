#!/usr/bin/env python3
"""Exact rectangle/mask and property checks for Cutout and TotalDropout."""
import pathlib
import subprocess
import sys
import tempfile


def cutout_reference(image, width, height, channels, rectangles, fill):
    out = bytearray(image)
    for x0, y0, x1, y1 in rectangles:
        for y in range(y0, y1):
            for x in range(x0, x1):
                p = (y * width + x) * channels
                out[p:p + channels] = bytes([fill]) * channels
    return bytes(out)


def run(exe, command, image, width, height, channels, args, td, name):
    source = pathlib.Path(td) / (name + "-in.raw")
    target = pathlib.Path(td) / (name + "-out.raw")
    source.write_bytes(bytes(image))
    subprocess.run([exe, *command, str(source), str(target), str(width), str(height), str(channels), *map(str, args)], check=True)
    return target.read_bytes()


def main():
    exe = sys.argv[1]
    width, height, channels = 6, 4, 3
    image = bytes(range(width * height * channels))
    rectangles = [(0, 1, 2, 3), (3, 0, 6, 1), (1, 2, 5, 4)]
    spec = ",".join(":".join(map(str, r)) for r in rectangles)
    with tempfile.TemporaryDirectory() as td:
        expected = cutout_reference(image, width, height, channels, rectangles, 231)
        assert run(exe, ["cutout"], image, width, height, channels, [231, spec], td, "cutout") == expected
        assert run(exe, ["dropout", "cutout"], image, width, height, channels, [231, spec], td, "nested") == expected
        assert run(exe, ["cutout"], image, width, height, channels, [17, "-"], td, "identity") == image
        assert run(exe, ["dropout", "total"], image, width, height, channels, [0.0, 19, 73], td, "keep") == image
        assert run(exe, ["dropout", "total"], image, width, height, channels, [1.0, 19, 73], td, "drop") == bytes([19]) * len(image)
        # An explicit mask is authoritative and makes probability/seed irrelevant.
        assert run(exe, ["dropout", "total"], image, width, height, channels, [19, 0, 0.0, 999], td, "mask-keep") == image
        assert run(exe, ["dropout", "total"], image, width, height, channels, [19, 1, 0.0, 999], td, "mask-drop") == bytes([19]) * len(image)
        assert run(exe, ["dropout", "total"], image, width, height, channels, [19, 1, 0.0, 999], td, "repeat") == bytes([19]) * len(image)
    print("PASS: Cutout rectangles and TotalDropout masks/properties")


if __name__ == "__main__":
    main()
