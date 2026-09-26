#!/usr/bin/env python3
"""Exact deterministic float32-reference checks for the tone contrast batch."""
import math
import pathlib
import struct
import subprocess
import sys
import tempfile


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference(image, operation, first, second):
    result = bytearray()
    for value in image:
        x = f32(f32(value) / f32(255.0))
        if operation == "sigmoid_contrast":
            exponent = f32(f32(second) * f32(f32(first) - x))
            exponential = f32(math.exp(exponent))
            y = f32(f32(1.0) / f32(f32(1.0) + exponential))
        else:
            scale = f32(f32(math.pow(f32(second), f32(first))) - f32(1.0))
            denominator = f32(math.log(f32(second)))
            y = f32(f32(math.log1p(f32(x * scale))) / denominator)
        scaled = f32(y * f32(255.0))
        result.append(max(0, min(255, int(math.floor(f32(scaled) + f32(0.5))))))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 19, 7, 4
image = bytes((i * 73 + i // 7 * 11 + 3) & 255 for i in range(width * height * channels))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source = root / "input.raw"
    source.write_bytes(image)
    cases = [
        ("sigmoid_contrast", 0.37, 8.25),
        ("sigmoid_contrast", 0.50, 0.0),
        ("log_contrast", 1.65, 2.0),
        ("log_contrast", 0.75, 3.0),
    ]
    for operation, first, second in cases:
        expected = reference(image, operation, first, second)
        output = root / (operation + str(first) + str(second) + ".raw")
        subprocess.run([exe, "tone", operation, str(source), str(output), str(width), str(height), str(channels), str(first), str(second)], check=True)
        actual = output.read_bytes()
        if actual != expected:
            raise AssertionError(f"{operation} float reference mismatch for {first}, {second}")
print("PASS: SigmoidContrast and LogContrast exact float-reference parity")
