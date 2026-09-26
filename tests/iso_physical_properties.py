#!/usr/bin/env python3
"""Property checks for physical-unit ISO dark-current and thermal noise APIs."""
import os
import struct
import subprocess
import sys
import tempfile


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack("<" + "f" * len(values), *values))


def read_f32(path, count):
    with open(path, "rb") as f:
        data = f.read()
    return struct.unpack("<" + "f" * count, data)


def run(exe, args):
    subprocess.run([exe] + args, check=True)


def main():
    exe = sys.argv[1]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.f32")
        knots = os.path.join(d, "knots.f32")
        out0 = os.path.join(d, "out0.f32")
        out1 = os.path.join(d, "out1.f32")
        out2 = os.path.join(d, "out2.f32")
        write_f32(src, [0.1, 0.2, 0.3, 0.4])
        # IsoNoiseProfilePoint's stable seven-float wire format.
        write_f32(knots, [100.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                           800.0, 4.0, 2.0, 3.0, 5.0, 0.0, 1.0])
        base = ["4", "1", "1", "450", knots, "2"]
        run(exe, ["iso_dark_current", src, out0] + base + ["100", "0", "1000", "91"])
        assert all(abs(a - b) < 1e-7 for a, b in zip(read_f32(out0, 4), (0.1, 0.2, 0.3, 0.4)))
        run(exe, ["iso_dark_current", src, out1] + base + ["100", "1", "1000", "91"])
        run(exe, ["iso_dark_current", src, out2] + base + ["100", "2", "1000", "91"])
        first, second = read_f32(out1, 4), read_f32(out2, 4)
        assert first == read_f32(out1, 4)
        assert all(0.0 <= x <= 1.0 for x in first + second)
        assert all(b >= a for a, b in zip(first, second))
        run(exe, ["iso_temperature_noise", src, out0] + base + ["0", "45", "25", "0.069314718", "1000", "7"])
        assert all(abs(a - b) < 1e-7 for a, b in zip(read_f32(out0, 4), (0.1, 0.2, 0.3, 0.4)))
        run(exe, ["iso_temperature_noise", src, out1] + base + ["20", "45", "25", "0.069314718", "1000", "7"])
        run(exe, ["iso_temperature_noise", src, out2] + base + ["20", "45", "25", "0.069314718", "1000", "7"])
        assert read_f32(out1, 4) == read_f32(out2, 4)
        assert all(0.0 <= x <= 1.0 for x in read_f32(out1, 4))
    print("PASS: ISO physical-unit dark-current and temperature properties")


if __name__ == "__main__":
    main()
