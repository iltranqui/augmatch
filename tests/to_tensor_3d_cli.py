#!/usr/bin/env python3
"""Property checks for the raw DHWC-to-CDHW ToTensor3D CLI contract."""
import pathlib
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
width, height, depth, channels = 3, 2, 2, 4
image = bytes((i * 17 + 3) % 256 for i in range(width * height * depth * channels))

def expected(normalize=False):
    means = (0.1, 0.2, 0.3)
    stds = (0.5, 0.25, 0.2)
    values = []
    for channel in range(channels):
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    source = (((z * height + y) * width + x) * channels + channel)
                    value = image[source] / 255.0
                    if normalize:
                        value = (value - means[min(channel, 2)]) / stds[min(channel, 2)]
                    values.append(value)
    return values

with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    source = directory / "input.raw"
    output = directory / "output.f32"
    source.write_bytes(image)
    for args, values in (
        ([], expected()),
        (["1", "0.1", "0.2", "0.3", "0.5", "0.25", "0.2"], expected(True)),
    ):
        subprocess.run(
            [exe, "to_tensor_3d", str(source), str(output), str(width), str(height), str(depth), str(channels)] + args,
            check=True,
        )
        raw = output.read_bytes()
        actual = struct.unpack("<%df" % len(values), raw)
        for index, (got, want) in enumerate(zip(actual, values)):
            if abs(got - want) > 1e-6:
                raise AssertionError((index, got, want))
print("PASS: ToTensor3D CLI DHWC uint8 to CDHW float32 properties")
