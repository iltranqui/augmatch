#!/usr/bin/env python3
"""Exact-mask and property checks for the deterministic MaskDropout contract."""
import pathlib
import subprocess
import sys
import tempfile


def reference(image, mask, width, height, channels, fill):
    out = bytearray(image)
    for y in range(height):
        for x in range(width):
            if mask[y * width + x] != 0:
                start = (y * width + x) * channels
                out[start:start + channels] = bytes([fill]) * channels
    return bytes(out)


def run(exe, image, mask, width, height, channels, fill, td):
    source = pathlib.Path(td) / "image.raw"
    mask_path = pathlib.Path(td) / "mask.raw"
    target = pathlib.Path(td) / "output.raw"
    source.write_bytes(bytes(image))
    mask_path.write_bytes(bytes(mask))
    subprocess.run([exe, "dropout", "mask", str(source), str(target), str(mask_path),
                    str(width), str(height), str(channels), str(fill)], check=True)
    return target.read_bytes()


def main():
    exe = sys.argv[1]
    width, height, channels = 3, 2, 2
    image = bytes(range(width * height * channels))
    mask = bytes([0, 1, 0, 255, 0, 1])
    expected = reference(image, mask, width, height, channels, 231)
    with tempfile.TemporaryDirectory() as td:
        got = run(exe, image, mask, width, height, channels, 231, td)
        assert got == expected, (list(got), list(expected))
        assert run(exe, image, [0] * (width * height), width, height, channels, 17, td) == image
        assert run(exe, image, [1] * (width * height), width, height, channels, 17, td) == bytes([17]) * len(image)
        # Nonzero mask bytes are equivalent, and repeated calls are deterministic.
        assert run(exe, image, [2, 0, 7, 0, 255, 0], width, height, channels, 99, td) == reference(image, [2, 0, 7, 0, 255, 0], width, height, channels, 99)
        assert run(exe, image, mask, width, height, channels, 231, td) == got
    print("PASS: MaskDropout exact-mask and property checks")


if __name__ == "__main__":
    main()
