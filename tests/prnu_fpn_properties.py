#!/usr/bin/env python3
"""Properties for seeded and explicit-map PRNU/FPN sensor operations."""
import pathlib
import statistics
import struct
import subprocess
import sys
import tempfile


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def run(exe, operation, values, width, height, channels, stddev, seed, mapping=None):
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        source, target = root / "in.f32", root / "out.f32"
        source.write_bytes(struct.pack("<%sf" % len(values), *values))
        args = [exe, operation, str(source), str(target), str(width), str(height), str(channels), str(stddev), str(seed)]
        if mapping is not None:
            map_path = root / "map.f32"
            map_path.write_bytes(struct.pack("<%sf" % len(mapping), *mapping))
            args.append(str(map_path))
        subprocess.run(args, check=True)
        return struct.unpack("<%sf" % len(values), target.read_bytes())


def main():
    exe = sys.argv[1]
    width, height, channels = 8, 7, 3
    count = width * height * channels
    signal = tuple(f32(0.25 + (i % 11) * 0.01) for i in range(count))
    gains = tuple(1.0 + ((i % 5) - 2) * 0.01 for i in range(count))
    offsets = tuple(((i % 7) - 3) * 0.002 for i in range(count))
    prnu_map = run(exe, "prnu", signal, width, height, channels, 0.0, 11, gains)
    fpn_map = run(exe, "fpn", signal, width, height, channels, 0.0, 11, offsets)
    for actual, expected in zip(prnu_map, (f32(f32(a) * f32(b)) for a, b in zip(signal, gains))):
        if actual != expected:
            raise AssertionError("explicit PRNU gain map was not applied exactly")
    for actual, expected in zip(fpn_map, (f32(f32(a) + f32(b)) for a, b in zip(signal, offsets))):
        if actual != expected:
            raise AssertionError("explicit FPN offset map was not applied exactly")
    if run(exe, "prnu", signal, width, height, channels, 0.04, 919) != run(exe, "prnu", signal, width, height, channels, 0.04, 919):
        raise AssertionError("seeded PRNU is not deterministic")
    if run(exe, "fpn", signal, width, height, channels, 0.02, 919) != run(exe, "fpn", signal, width, height, channels, 0.02, 919):
        raise AssertionError("seeded FPN is not deterministic")
    zero_prnu = run(exe, "prnu", signal, width, height, channels, 0.0, 919)
    zero_fpn = run(exe, "fpn", signal, width, height, channels, 0.0, 919)
    if zero_prnu != signal or zero_fpn != signal:
        raise AssertionError("zero-strength sensor noise is not identity")
    flat = (0.5,) * (128 * 128)
    prnu_sample = run(exe, "prnu", flat, 128, 128, 1, 0.05, 77)
    high_prnu_sample = run(exe, "prnu", (1.0,) * (128 * 128), 128, 128, 1, 0.05, 77)
    fpn_sample = run(exe, "fpn", flat, 128, 128, 1, 0.03, 77)
    if not 0.49 < statistics.mean(prnu_sample) < 0.51 or not 0.02 < statistics.stdev(prnu_sample) < 0.04:
        raise AssertionError("PRNU does not scale with signal at the expected standard deviation")
    if not 1.8 < statistics.stdev(high_prnu_sample) / statistics.stdev(prnu_sample) < 2.2:
        raise AssertionError("PRNU variation did not scale linearly with signal")
    if abs(statistics.mean(fpn_sample) - 0.5) > 0.003 or not 0.02 < statistics.stdev(fpn_sample) < 0.04:
        raise AssertionError("FPN offset statistics are outside the expected range")
    print("PASS: deterministic explicit-map and seeded PRNU/FPN properties")


if __name__ == "__main__":
    main()
