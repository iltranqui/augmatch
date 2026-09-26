#!/usr/bin/env python3
"""CLI identity and deterministic checks for the eight catalog weather APIs."""
import pathlib
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
w, h, c = 4, 3, 3
image = bytes((17 + i * 13) % 256 for i in range(w * h * c))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "input.raw"
    source.write_bytes(image)

    def run(*args):
        output = root / ("out-%d.raw" % len(list(root.glob("out-*.raw"))))
        subprocess.run([exe, *map(str, args[:1]), str(source), str(output), str(w), str(h), str(c), *map(str, args[1:])], check=True)
        return output.read_bytes()

    assert run("fast_snowy_landscape", .5, 0) == image
    first = run("clouds", .4, .8, 99)
    second = run("clouds", .4, .8, 99)
    assert first == second
    assert run("fog", 1, 0, 7) == image

    layer = root / "layer.f32"
    layer.write_bytes(struct.pack("<%df" % (w * h), *([0.0] * (w * h))))
    assert run("cloud_layer", 1, layer) == image
    assert run("snowflakes_layer", 1, layer) == image
    assert run("rain_layer", 1, layer) == image

    assert run("snowflakes", 0, .5, 4) == image
    assert run("rain", 0, .5, 4) == image
print("PASS: weather catalog deterministic and identity properties")
