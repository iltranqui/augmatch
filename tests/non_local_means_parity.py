#!/usr/bin/env python3
"""Direct deterministic reference for the bounded NonLocalMeansDenoising contract."""
import pathlib
import subprocess
import sys
import tempfile
import math


def reflect101(value, size):
    if size == 1:
        return 0
    while value < 0 or value >= size:
        value = -value if value < 0 else 2 * size - value - 2
    return value


def reference(image, width, height, channels, patch_radius, search_radius, h):
    output = bytearray(len(image))
    normalizer = (2 * patch_radius + 1) ** 2 * channels
    inv_h2 = 1.0 / (h * h)
    for y in range(height):
        for x in range(width):
            sums = [0.0] * channels
            weight_sum = 0.0
            for sy in range(-search_radius, search_radius + 1):
                for sx in range(-search_radius, search_radius + 1):
                    distance = 0.0
                    for py in range(-patch_radius, patch_radius + 1):
                        for px in range(-patch_radius, patch_radius + 1):
                            ax = reflect101(x + px, width)
                            ay = reflect101(y + py, height)
                            bx = reflect101(x + sx + px, width)
                            by = reflect101(y + sy + py, height)
                            a = (ay * width + ax) * channels
                            b = (by * width + bx) * channels
                            for channel in range(channels):
                                difference = image[a + channel] - image[b + channel]
                                distance += difference * difference
                    weight = math.exp(-(distance / normalizer) * inv_h2)
                    sample_x = reflect101(x + sx, width)
                    sample_y = reflect101(y + sy, height)
                    sample = (sample_y * width + sample_x) * channels
                    for channel in range(channels):
                        sums[channel] += weight * image[sample + channel]
                    weight_sum += weight
            output[(y * width + x) * channels:(y * width + x + 1) * channels] = bytes(
                int(max(0.0, min(255.0, sums[channel] / weight_sum)) + 0.5)
                for channel in range(channels)
            )
    return output


def main():
    executable = sys.argv[1]
    width, height, channels = 5, 4, 2
    image = bytes((17 * i + 31 * (i // channels) + i % channels) % 256
                  for i in range(width * height * channels))
    patch_radius, search_radius, h = 1, 2, 18.0
    expected = reference(image, width, height, channels, patch_radius, search_radius, h)
    with tempfile.TemporaryDirectory() as directory:
        source = pathlib.Path(directory) / "input.raw"
        destination = pathlib.Path(directory) / "output.raw"
        source.write_bytes(image)
        subprocess.run([executable, "non_local_means", str(source), str(destination),
                        str(width), str(height), str(channels), str(patch_radius),
                        str(search_radius), str(h)], check=True)
        actual = destination.read_bytes()
    if actual != expected:
        raise AssertionError("NonLocalMeansDenoising output differs from direct reference")
    print("PASS: NonLocalMeansDenoising deterministic direct reference parity")


if __name__ == "__main__":
    main()
