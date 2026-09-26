#!/usr/bin/env python3
"""Reference parity for the explicit RandomGridShuffle permutation/seed contract."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def permutation(rows, cols, seed):
    result = list(range(rows * cols))
    for i in range(len(result) - 1, 0, -1):
        j = splitmix64((seed + i) & ((1 << 64) - 1)) % (i + 1)
        result[i], result[j] = result[j], result[i]
    return result


def reference(image, width, height, channels, rows, cols, cell_permutation):
    output = bytearray(len(image))
    for y in range(height):
        for x in range(width):
            destination_row = y * rows // height
            destination_col = x * cols // width
            destination = destination_row * cols + destination_col
            source = cell_permutation[destination]
            dx0, dx1 = destination_col * width // cols, (destination_col + 1) * width // cols
            dy0, dy1 = destination_row * height // rows, (destination_row + 1) * height // rows
            source_col, source_row = source % cols, source // cols
            sx0, sx1 = source_col * width // cols, (source_col + 1) * width // cols
            sy0, sy1 = source_row * height // rows, (source_row + 1) * height // rows
            sx = sx0 + min(sx1 - sx0 - 1, (x - dx0) * (sx1 - sx0) // (dx1 - dx0))
            sy = sy0 + min(sy1 - sy0 - 1, (y - dy0) * (sy1 - sy0) // (dy1 - dy0))
            for channel in range(channels):
                output[(y * width + x) * channels + channel] = image[(sy * width + sx) * channels + channel]
    return output


exe = sys.argv[1]
width, height, channels, rows, cols, seed = 10, 7, 2, 3, 4, 0x123456789ABCDEF0
image = bytes((i * 17 + 3) % 256 for i in range(width * height * channels))
seeded = permutation(rows, cols, seed)
explicit = list(reversed(range(rows * cols)))
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source, output, repeat = root / "input.raw", root / "output.raw", root / "repeat.raw"
    source.write_bytes(image)
    command = [exe, "random_grid_shuffle", str(source), str(output), str(width), str(height),
               str(channels), str(rows), str(cols), str(seed)]
    subprocess.run(command, check=True)
    expected = reference(image, width, height, channels, rows, cols, seeded)
    actual = output.read_bytes()
    if actual != expected:
        raise AssertionError("seeded RandomGridShuffle reference mismatch")
    subprocess.run(command[:-1] + [str(seed)], check=True)
    if output.read_bytes() != actual:
        raise AssertionError("seeded RandomGridShuffle is not deterministic")
    permutation_file = root / "permutation.u32"
    permutation_file.write_bytes(b"".join(struct.pack("<I", value) for value in explicit))
    subprocess.run(command + [str(permutation_file)], check=True)
    expected_explicit = reference(image, width, height, channels, rows, cols, explicit)
    if output.read_bytes() != expected_explicit:
        raise AssertionError("explicit RandomGridShuffle permutation mismatch")
print("PASS: RandomGridShuffle seed and explicit permutation reference parity")
