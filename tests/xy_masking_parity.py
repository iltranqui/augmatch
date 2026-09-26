#!/usr/bin/env python3
"""Exact-mask and property checks for deterministic XYMasking."""
import pathlib
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, rows, columns, fill):
    out = bytearray(image)
    for y in range(height):
        for x in range(width):
            masked = any(begin <= y < end for begin, end in rows)
            masked = masked or any(begin <= x < end for begin, end in columns)
            if masked:
                start = (y * width + x) * channels
                out[start:start + channels] = bytes([fill]) * channels
    return bytes(out)


def run(exe, image, width, height, channels, fill, rows, columns, td, nested=False):
    source = pathlib.Path(td) / "image.raw"
    target = pathlib.Path(td) / ("nested.raw" if nested else "output.raw")
    source.write_bytes(bytes(image))
    row_spec = ",".join(f"{begin}:{end}" for begin, end in rows) or "-"
    column_spec = ",".join(f"{begin}:{end}" for begin, end in columns) or "-"
    command = [exe]
    if nested:
        command += ["dropout", "xy_masking"]
    else:
        command += ["xy_masking"]
    command += [str(source), str(target), str(width), str(height), str(channels),
                str(fill), row_spec, column_spec]
    subprocess.run(command, check=True)
    return target.read_bytes()


def main():
    exe = sys.argv[1]
    width, height, channels = 5, 4, 3
    image = bytes(range(width * height * channels))
    rows = [(1, 2), (3, 4)]
    columns = [(0, 1), (3, 5)]
    with tempfile.TemporaryDirectory() as td:
        expected = reference(image, width, height, channels, rows, columns, 231)
        got = run(exe, image, width, height, channels, 231, rows, columns, td)
        assert got == expected, (list(got), list(expected))
        # Empty intervals are an identity operation and a row/column union masks
        # every selected pixel, independent of the interval order or overlap.
        assert run(exe, image, width, height, channels, 17, [], [], td) == image
        all_rows = [(0, height)]
        assert run(exe, image, width, height, channels, 17, all_rows, [], td) == bytes([17]) * len(image)
        assert run(exe, image, width, height, channels, 17, [], [(0, width)], td) == bytes([17]) * len(image)
        assert run(exe, image, width, height, channels, 231, rows, columns, td) == got
        assert run(exe, image, width, height, channels, 231, rows, columns, td, nested=True) == expected
    print("PASS: XYMasking exact-interval and property checks")


if __name__ == "__main__":
    main()
