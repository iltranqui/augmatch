#!/usr/bin/env python3
import pathlib
import statistics
import subprocess
import sys
import tempfile


def run(exe, root, image, width, height, sigma, gradient, laplacian, seed):
    source = root / "input.raw"
    output = root / "output.raw"
    source.write_bytes(image)
    subprocess.run([exe, "gradient_plus_laplacian_noise", str(source), str(output),
                    str(width), str(height), "1", str(sigma), str(gradient),
                    str(laplacian), str(seed)], check=True)
    return list(output.read_bytes())


def main():
    exe = sys.argv[1]
    width = height = 64
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        flat = bytes([128]) * (width * height)
        sample = run(exe, root, flat, width, height, 12, 0, 0, 4)
        assert abs(statistics.mean(sample) - 128) < 2.5
        assert 8 < statistics.pstdev(sample) < 16
        assert run(exe, root, flat, width, height, 12, 0, 0, 4) == sample

        edge = bytes((32 if x < width // 2 else 224) for y in range(height) for x in range(width))
        baseline = run(exe, root, edge, width, height, 8, 0, 0, 9)
        amplified = run(exe, root, edge, width, height, 8, 8, 8, 9)
        boundary = [y * width + x for y in range(height) for x in (width // 2 - 1, width // 2)]
        assert sum(abs(amplified[i] - edge[i]) for i in boundary) > sum(abs(baseline[i] - edge[i]) for i in boundary)
        assert all(0 <= value <= 255 for value in amplified)
    print("gradient-plus-Laplacian noise properties passed")


if __name__ == "__main__":
    main()
