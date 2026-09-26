#!/usr/bin/env python3
"""Reference parity for the deterministic regular-grid Superpixels contract."""
import pathlib
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, cell_width, cell_height):
    output = bytearray(len(image))
    for y in range(height):
        for x in range(width):
            x0 = (x // cell_width) * cell_width
            y0 = (y // cell_height) * cell_height
            x1 = min(width, x0 + cell_width)
            y1 = min(height, y0 + cell_height)
            count = (x1 - x0) * (y1 - y0)
            for channel in range(channels):
                total = sum(
                    image[(py * width + px) * channels + channel]
                    for py in range(y0, y1)
                    for px in range(x0, x1)
                )
                output[(y * width + x) * channels + channel] = (total + count // 2) // count
    return bytes(output)


exe = sys.argv[1]
width, height, channels = 7, 5, 3
cell_width, cell_height = 3, 2
image = bytes((i * 37 + 13) % 256 for i in range(width * height * channels))
expected = reference(image, width, height, channels, cell_width, cell_height)
with tempfile.TemporaryDirectory() as td:
    source = pathlib.Path(td) / "in.raw"
    destination = pathlib.Path(td) / "out.raw"
    source.write_bytes(image)
    subprocess.run(
        [exe, "superpixels", str(source), str(destination), str(width), str(height),
         str(channels), str(cell_width), str(cell_height)],
        check=True,
    )
    got = destination.read_bytes()

if got != expected:
    raise AssertionError("Superpixels output differs from deterministic regular-grid reference")
print("PASS: Superpixels deterministic regular-grid parity")
