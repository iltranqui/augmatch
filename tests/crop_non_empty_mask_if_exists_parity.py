#!/usr/bin/env python3
"""Reference checks for the deterministic image+mask target-aware crop."""
import pathlib
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1

def splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) & MASK

def rectangle(mask, w, h, cw, ch, seed, fx, fy):
    targets = [(x, y) for y in range(h) for x in range(w) if mask[y * w + x] != 0]
    if not targets:
        x = fx if fx >= 0 else splitmix64(seed) % (w - cw + 1)
        y = fy if fy >= 0 else splitmix64(seed + 1) % (h - ch + 1)
        return x, y, False
    tx, ty = targets[splitmix64(seed) % len(targets)]
    min_x, max_x = max(0, tx - cw + 1), min(tx, w - cw)
    min_y, max_y = max(0, ty - ch + 1), min(ty, h - ch)
    x = min_x + splitmix64(seed + 1) % (max_x - min_x + 1)
    y = min_y + splitmix64(seed + 2) % (max_y - min_y + 1)
    return x, y, True

def run(exe, image, mask, w, h, c, mc, cw, ch, seed, fx, fy, td, threshold=None):
    p = pathlib.Path(td)
    ip, mp, op, omp = p / "image.raw", p / "mask.raw", p / "out.raw", p / "out_mask.raw"
    ip.write_bytes(image); mp.write_bytes(mask)
    args = [exe, "crop_non_empty_mask_if_exists", str(ip), str(mp), str(op), str(omp), str(w), str(h), str(c), str(mc), str(cw), str(ch), str(seed), str(fx), str(fy)]
    if threshold is not None: args.append(str(threshold))
    subprocess.run(args, check=True)
    return op.read_bytes(), omp.read_bytes()

def main():
    exe = str(pathlib.Path(__file__).resolve().parents[1] / "build-crop-cpu" / "augmatch_cli") if len(sys.argv) < 2 else sys.argv[1]
    w, h, c, mc, cw, ch = 11, 9, 2, 1, 5, 4
    image = bytes((i * 17 + 3) & 255 for i in range(w * h * c))
    mask = bytearray(w * h); mask[1 * w + 8] = 3; mask[7 * w + 2] = 9
    with tempfile.TemporaryDirectory() as td:
        for seed in (0, 1, 77, 0x123456789ABCDEF0):
            actual, actual_mask = run(exe, image, mask, w, h, c, mc, cw, ch, seed, -1, -1, td)
            x, y, used = rectangle(mask, w, h, cw, ch, seed, -1, -1)
            assert used and any(mask[(y + row) * w + x:(y + row) * w + x + cw] for row in range(ch))
            expected = b"".join(image[((y + row) * w + x) * c:((y + row) * w + x + cw) * c] for row in range(ch))
            expected_mask = b"".join(mask[(y + row) * w + x:(y + row) * w + x + cw] for row in range(ch))
            assert actual == expected and actual_mask == expected_mask
            assert run(exe, image, mask, w, h, c, mc, cw, ch, seed, -1, -1, td) == (actual, actual_mask)
        empty = bytes(w * h)
        actual, actual_mask = run(exe, image, empty, w, h, c, mc, cw, ch, 999, 3, 2, td)
        expected = b"".join(image[((2 + row) * w + 3) * c:((2 + row) * w + 3 + cw) * c] for row in range(ch))
        assert actual == expected and actual_mask == bytes(cw * ch)
    print("PASS: CropNonEmptyMaskIfExists deterministic target selection, fallback, and image/mask outputs")

if __name__ == "__main__":
    main()
