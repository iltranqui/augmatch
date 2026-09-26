#!/usr/bin/env python3
"""Exact vectors and invariants for imgaug uniform color quantizers."""
import math
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
w, h, c = 7, 3, 4
values = bytes((i * 37 + 11) % 256 for i in range(w * h * c))

def center_quantize(v, levels):
    if levels == 256:
        return v
    q = 256.0 / levels
    return max(0, min(255, int(math.floor(math.floor(v / q) * q + q / 2.0 + 0.5))))

def bits_quantize(v, bits):
    return v & (0xff << (8 - bits))

def run(op, arg, source, target):
    subprocess.run([exe, "color", op, str(source), str(target), str(w), str(h), str(c), str(arg)], check=True)
    return target.read_bytes()

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    source = td / "input.raw"
    source.write_bytes(values)
    for levels in (2, 3, 6, 17, 128, 256):
        target = td / ("levels-%d.raw" % levels)
        got = run("uniform_color_quantization", levels, source, target)
        expected = bytes(
            value if channel == 3 else center_quantize(value, levels)
            for index, value in enumerate(values)
            for channel in [index % c]
        )
        assert got == expected, "levels=%d mismatch" % levels
        assert all(0 <= value <= 255 for value in got)
        assert len(set(got[0::c])) <= levels
    for bits in (1, 2, 3, 5, 8):
        target = td / ("bits-%d.raw" % bits)
        got = run("uniform_color_quantization_to_n_bits", bits, source, target)
        expected = bytes(
            value if channel == 3 else bits_quantize(value, bits)
            for index, value in enumerate(values)
            for channel in [index % c]
        )
        assert got == expected, "bits=%d mismatch" % bits
        assert all((got[index] == values[index]) if index % c == 3 else
                   (got[index] & ((1 << (8 - bits)) - 1)) == 0
                   for index in range(len(values)))
    # CLI aliases must preserve the same exact output.
    alias = td / "alias.raw"
    assert run("quantize_uniform", 6, source, alias) == run("uniform_color_quantization", 6, source, td / "canonical.raw")
    assert run("quantize_uniform_to_n_bits", 3, source, alias) == run("uniform_color_quantization_to_n_bits", 3, source, td / "canonical-bits.raw")
print("PASS: uniform color quantization exact vectors and properties")
