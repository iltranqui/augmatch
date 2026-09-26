#!/usr/bin/env python3
"""Verify the CLI's little-endian CHW float32 binary contract without NumPy."""
import pathlib
import struct
import subprocess
import sys
import tempfile

exe = sys.argv[1]
image = bytes(range(1, 13))
expected = [1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]
with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    source = directory / "input.raw"
    output = directory / "output.f32"
    source.write_bytes(image)
    subprocess.run([exe, "to_tensor_v2", str(source), str(output), "2", "2", "3"], check=True)
    values = struct.unpack("<12f", output.read_bytes())
    for actual, value in zip(values, expected):
        if abs(actual - value / 255.0) > 1e-7:
            raise AssertionError((actual, value))
print("PASS: ToTensorV2 CLI CHW float32 contract")
