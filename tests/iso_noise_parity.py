#!/usr/bin/env python3
"""Independent scalar reference for the explicit ISONoise contract."""
import math
import pathlib
import statistics
import struct
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def mix(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK

def uniform(x):
    return f32((mix(x) >> 11) * (1.0 / 9007199254740992.0))

def normal(x):
    a = max(uniform(x), 1.17549435e-38)
    return f32(f32(math.sqrt(f32(-2.0 * math.log(a)))) *
                f32(math.cos(f32(6.28318530718 * uniform(x ^ 0xD1B54A32D192ED03)))))

def key(seed, x, y, channel, tag):
    return (seed ^ ((x * 0x632BE59BD9B4E019) & MASK) ^
            ((y * 0x8CB92BAA2F2F6F7D) & MASK) ^
            ((channel * 0x9E3779B97F4A7C15) & MASK) ^ tag) & MASK

def reference(image, width, height, channels, iso, base_iso, analog, digital, gaussian, chroma, seed):
    gain = f32(f32(iso / base_iso) * analog)
    weights = (f32(0.2126), f32(0.7152), f32(0.0722))
    result = []
    for y in range(height):
        for x in range(width):
            zsum = f32(0.0)
            for ch in range(min(channels, 3)):
                zsum = f32(zsum + f32(weights[ch] * normal(key(seed, x, y, ch, 0x49534F5F4348524F))))
            center = zsum if channels >= 3 else f32(0.0)
            luma = normal(key(seed, x, y, 0, 0x49534F5F4C554D41))
            for ch in range(channels):
                z = normal(key(seed, x, y, ch, 0x49534F5F4348524F))
                color_noise = f32(z - center) if channels >= 3 else f32(0.0)
                value = f32(f32(f32(image[(y * width + x) * channels + ch] / 255.0) * digital) +
                            f32(f32(gaussian * gain) * luma) +
                            f32(f32(chroma * gain) * color_noise))
                value = max(0.0, min(1.0, value))
                result.append(int(math.floor(f32(value * 255.0 + 0.5))))
    return bytes(result)

def run(exe, image, args, width=3, height=2, channels=3):
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "in.raw"
        dst = pathlib.Path(td) / "out.raw"
        src.write_bytes(image)
        subprocess.run([exe, "iso_noise", str(src), str(dst), str(width), str(height), str(channels), *map(str, args)], check=True)
        return dst.read_bytes()

def main():
    exe = sys.argv[1]
    image = bytes((0, 31, 127, 191, 240, 255, 17, 88, 144, 62, 203, 229, 255, 2, 76, 119, 181, 222))
    args = (800.0, 100.0, 1.25, 0.9, 0.035, 0.02, 918273645)
    actual = run(exe, image, args)
    expected = reference(image, 3, 2, 3, *args)
    if actual != expected:
        raise AssertionError("ISONoise output differs from independent hash/RNG reference")
    if run(exe, image, args) != actual:
        raise AssertionError("ISONoise is not deterministic for a fixed seed")
    clean = run(exe, image, (100.0, 100.0, 1.0, 1.0, 0.0, 0.0, 918273645))
    if clean != image:
        raise AssertionError("zero-noise identity property failed")
    sample = run(exe, bytes([128]) * (64 * 64), (100.0, 100.0, 1.0, 1.0, 0.03, 0.0, 77), 64, 64, 1)
    values = list(sample)
    if not 120.0 < statistics.mean(values) < 136.0 or not 5.0 < statistics.stdev(values) < 10.0:
        raise AssertionError("Gaussian ISO noise statistics are outside the contract range")
    print("PASS: ISO/gain Gaussian/chroma noise parity and properties")

if __name__ == "__main__":
    main()
