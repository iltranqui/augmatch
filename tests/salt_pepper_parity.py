#!/usr/bin/env python3
"""Exact explicit-mask/rectangle and deterministic seeded salt-pepper checks."""
import pathlib
import subprocess
import sys
import tempfile


def run(exe, op, image, w, h, c, args, td, name):
    source = pathlib.Path(td) / (name + "-in.raw")
    target = pathlib.Path(td) / (name + "-out.raw")
    source.write_bytes(image)
    subprocess.run([exe, "arithmetic", op, str(source), str(target), str(w), str(h), str(c), *map(str, args)], check=True)
    return target.read_bytes()


def main():
    exe = sys.argv[1]
    w, h, c = 7, 5, 3
    image = bytes([91] * (w * h * c))
    with tempfile.TemporaryDirectory() as td:
        mask_path = pathlib.Path(td) / "mask.raw"
        mask = bytearray(w * h)
        mask[1], mask[9] = 1, 1
        mask_path.write_bytes(mask)
        salt = run(exe, "salt", image, w, h, c, [mask_path], td, "salt-mask")
        for p in range(w * h):
            expected = 255 if mask[p] else 91
            assert salt[p*c:(p+1)*c] == bytes([expected] * c)
        pepper = run(exe, "pepper", image, w, h, c, [mask_path], td, "pepper-mask")
        for p in range(w * h):
            expected = 0 if mask[p] else 91
            assert pepper[p*c:(p+1)*c] == bytes([expected] * c)
        rect = "1:1:5:4,0:0:1:1"
        coarse = run(exe, "coarse_pepper", image, w, h, c, [rect], td, "pepper-rect")
        for y in range(h):
            for x in range(w):
                selected = (1 <= x < 5 and 1 <= y < 4) or (x == 0 and y == 0)
                p = (y*w+x)*c
                assert coarse[p:p+c] == bytes([0 if selected else 91] * c)
        a = run(exe, "coarse_salt", image, w, h, c, [0.4, 71, 2, 3], td, "seed-a")
        b = run(exe, "coarse_salt", image, w, h, c, [0.4, 71, 2, 3], td, "seed-b")
        assert a == b
        for y in range(h):
            for x in range(w):
                p = (y*w+x)*c
                assert len(set(a[p:p+c])) == 1 and a[p] in (91, 255)
    print("PASS: Salt/Pepper exact masks, rectangles, and seeded coarse properties")


if __name__ == "__main__":
    main()
