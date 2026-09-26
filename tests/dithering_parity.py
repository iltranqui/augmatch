#!/usr/bin/env python3
"""Exact and property checks for deterministic ordered and Floyd-Steinberg dithering."""
import pathlib
import random
import subprocess
import sys
import tempfile


BAYER = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))


def encoded(level, levels):
    return max(0, min(255, int(level * 255 / levels + 0.5)))


def ordered(image, width, height, channels, bits):
    levels = (1 << bits) - 1
    result = bytearray(len(image))
    for y in range(height):
        for x in range(width):
            threshold = (BAYER[y & 3][x & 3] + 0.5) / 16.0
            for channel in range(channels):
                index = (y * width + x) * channels + channel
                level = max(0, min(levels, int(image[index] / 255.0 * levels + threshold)))
                result[index] = encoded(level, levels)
    return result


def floyd_steinberg(image, width, height, channels, bits):
    levels = (1 << bits) - 1
    result = bytearray(len(image))
    for channel in range(channels):
        errors = [0.0] * (width * height)
        for y in range(height):
            for x in range(width):
                pixel = y * width + x
                index = pixel * channels + channel
                working = max(0.0, min(1.0, image[index] / 255.0 + errors[pixel]))
                level = max(0, min(levels, int(working * levels + 0.5)))
                result[index] = encoded(level, levels)
                error = working - level / levels
                if x + 1 < width:
                    errors[pixel + 1] += error * 7.0 / 16.0
                if y + 1 < height:
                    errors[pixel + width] += error * 5.0 / 16.0
                    if x > 0:
                        errors[pixel + width - 1] += error * 3.0 / 16.0
                    if x + 1 < width:
                        errors[pixel + width + 1] += error * 1.0 / 16.0
    return result


def run(cli, image, width, height, channels, bits, mode, nested=True):
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        source, target = directory / "input.raw", directory / "output.raw"
        source.write_bytes(image)
        if nested:
            command = [str(cli), "color", "dithering", str(source), str(target), str(width), str(height), str(channels), str(bits), mode]
        else:
            command = [str(cli), "dithering", str(source), str(target), str(width), str(height), str(channels), str(bits), mode]
        subprocess.run(command, check=True)
        return target.read_bytes()


def main():
    cli = pathlib.Path(sys.argv[1])
    width, height, channels = 13, 9, 3
    rng = random.Random(20250315)
    image = bytes(rng.randrange(256) for _ in range(width * height * channels))
    for bits in (1, 2, 4, 7, 8):
        actual = run(cli, image, width, height, channels, bits, "ordered")
        expected = ordered(image, width, height, channels, bits)
        if actual != expected:
            raise AssertionError(f"ordered mismatch at bit depth {bits}")
        actual = run(cli, image, width, height, channels, bits, "error_diffusion", nested=False)
        expected = floyd_steinberg(image, width, height, channels, bits)
        if actual != expected:
            raise AssertionError(f"error diffusion mismatch at bit depth {bits}")
        levels = (1 << bits) - 1
        allowed = {encoded(level, levels) for level in range(levels + 1)}
        if any(value not in allowed for value in actual):
            raise AssertionError(f"output contains a non-{bits}-bit level")
    if run(cli, image, width, height, channels, 8, "ordered") != image:
        raise AssertionError("8-bit ordered dithering must preserve input")
    if run(cli, image, width, height, channels, 8, "error_diffusion") != image:
        raise AssertionError("8-bit error diffusion must preserve input")
    print("Dithering exact/property parity passed")


if __name__ == "__main__":
    main()
