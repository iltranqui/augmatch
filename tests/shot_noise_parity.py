#!/usr/bin/env python3
"""Independent deterministic and distributional checks for ShotNoise."""
import math
import pathlib
import statistics
import struct
import subprocess
import sys
import tempfile

MASK = (1 << 64) - 1

def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]

def mix(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) & MASK

def uniform(value):
    return f32((mix(value) >> 11) * (1.0 / 9007199254740992.0))

def normal(value):
    a = max(uniform(value), f32(1.17549435e-38))
    return f32(f32(math.sqrt(f32(-2.0 * math.log(a)))) *
                f32(math.cos(f32(6.28318530718 * uniform(value ^ 0xD1B54A32D192ED03)))))

def key(seed, x, y, channel, tag):
    return (seed ^ ((x * 0x632BE59BD9B4E019) & MASK) ^
            ((y * 0x8CB92BAA2F2F6F7D) & MASK) ^
            ((channel * 0x9E3779B97F4A7C15) & MASK) ^ tag) & MASK

def poisson(rate, seed):
    if rate <= 0.0:
        return 0
    if rate < 64.0:
        u = uniform(seed ^ 0x53484F545F554E49)
        probability = f32(math.exp(-rate))
        cdf = probability
        count = 0
        while u > cdf and count < 512:
            count += 1
            probability = f32(probability * f32(rate / count))
            cdf = f32(cdf + probability)
        return count
    return int(math.floor(max(0.0, f32(rate + f32(math.sqrt(rate)) * normal(seed ^ 0x53484F545F4E4F52))) + 0.5))

def reference(image, width, height, channels, gain, scale, seed):
    gain, scale = f32(gain), f32(scale)
    result = []
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                i = (y * width + x) * channels + channel
                rate = f32(f32(f32(image[i] / 255.0) * gain) * scale)
                count = poisson(rate, key(seed, x, y, channel, 0x53484F545F4E4F49))
                value = f32(count / scale)
                value = max(0.0, min(1.0, value))
                result.append(int(math.floor(f32(value * 255.0 + 0.5))))
    return bytes(result)

def run(exe, image, args, width, height, channels):
    with tempfile.TemporaryDirectory() as td:
        source = pathlib.Path(td) / "in.raw"
        target = pathlib.Path(td) / "out.raw"
        source.write_bytes(image)
        subprocess.run([exe, "shot_noise", str(source), str(target), str(width), str(height), str(channels), *map(str, args)], check=True)
        return target.read_bytes()

def main():
    exe = sys.argv[1]
    width, height, channels = 7, 5, 3
    image = bytes((i * 37 + 11) % 256 for i in range(width * height * channels))
    args = (1.15, 32.0, 918273645)
    actual = run(exe, image, args, width, height, channels)
    expected = reference(image, width, height, channels, *args)
    if actual != expected:
        raise AssertionError("ShotNoise output differs from independent Poisson reference")
    if run(exe, image, args, width, height, channels) != actual:
        raise AssertionError("ShotNoise is not deterministic for a fixed seed")
    if run(exe, image, (0.0, 32.0, 7), width, height, channels) != bytes(len(image)):
        raise AssertionError("zero gain did not produce black output")
    sample = run(exe, bytes([128]) * (128 * 128), (1.0, 64.0, 77), 128, 128, 1)
    values = list(sample)
    if not 124.0 < statistics.mean(values) < 132.0:
        raise AssertionError("Poisson shot-noise mean is outside the contract range")
    if not 18.0 < statistics.stdev(values) < 27.0:
        raise AssertionError("Poisson shot-noise standard deviation is outside the contract range")
    print("PASS: deterministic Poisson ShotNoise parity and distribution checks")

if __name__ == "__main__":
    main()
