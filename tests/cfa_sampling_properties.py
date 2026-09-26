#!/usr/bin/env python3
"""Exact CLI vectors plus periodic-mask properties for the CFA samplers."""
import pathlib
import subprocess
import sys
import tempfile


def run(exe, operation, source, width, height, *args):
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "in.raw"
        dst = pathlib.Path(td) / "out.raw"
        src.write_bytes(bytes(source))
        subprocess.run([exe, operation, str(src), str(dst), str(width), str(height), *map(str, args)], check=True)
        return list(dst.read_bytes())


def main(exe):
    w, h = 9, 7
    rgb = [v for _ in range(w * h) for v in (10, 20, 30)]
    quad = run(exe, "quad_bayer", rgb, w, h, 0)
    expected = []
    table = ((0, 1), (1, 2))
    for y in range(h):
        for x in range(w):
            expected.append((10, 20, 30)[table[(y // 2) & 1][(x // 2) & 1]])
    assert quad == expected

    rgbw = [v for _ in range(w * h) for v in (11, 22, 33, 44)]
    rgbw_out = run(exe, "rgbw", rgbw, w, h, 0)
    expected = []
    for y in range(h):
        for x in range(w):
            expected.append((11, 22, 44, 33)[(y & 1) * 2 + (x & 1)])
    assert rgbw_out == expected

    mask = bytes((0, 1, 2, 3, 3, 2))
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        src, mask_path, dst = td / "in.raw", td / "mask.raw", td / "out.raw"
        channels = 4
        src.write_bytes(bytes(v for _ in range(w * h) for v in (1, 2, 3, 4)))
        mask_path.write_bytes(mask)
        subprocess.run([exe, "custom_cfa", str(src), str(dst), str(w), str(h), str(channels), "3", "2", str(mask_path)], check=True)
        actual = list(dst.read_bytes())
    expected = [mask[(y % 2) * 3 + (x % 3)] + 1 for y in range(h) for x in range(w)]
    assert actual == expected
    print("PASS: CFA exact vectors and periodic-mask properties")


if __name__ == "__main__":
    main(sys.argv[1])
