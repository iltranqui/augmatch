#!/usr/bin/env python3
"""Exact reference checks for the deterministic SnowStamp image/mask contract."""
import pathlib
import struct
import subprocess
import sys
import tempfile


def reference(image, width, height, channels, stamp, mask, sw, sh, placements, alpha, fill):
    result = bytearray(image)
    for x, y, local_alpha in placements:
        for sy in range(sh):
            ty = y + sy
            if ty < 0 or ty >= height:
                continue
            for sx in range(sw):
                tx = x + sx
                if tx < 0 or tx >= width:
                    continue
                a = max(0.0, min(1.0, alpha * local_alpha * mask[sy * sw + sx] / 255.0))
                if a == 0.0:
                    continue
                source = sy * sw + sx
                destination = (ty * width + tx) * channels
                for channel in range(channels):
                    value = (1.0 - a) * result[destination + channel] + a * stamp[source * channels + channel]
                    result[destination + channel] = max(0, min(255, int(value + 0.5)))
    return bytes(result)


exe = sys.argv[1]
width, height, channels = 5, 4, 3
sw, sh = 3, 2
image = bytes((i * 17 + 3) % 256 for i in range(width * height * channels))
stamp = bytes((i * 41 + 29) % 256 for i in range(sw * sh * channels))
mask = bytes((20, 128, 255, 255, 0, 64))
placements = ((-1, 1, 0.5), (1, 1, 1.0), (1, 1, 0.25))
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    source, stamp_file, mask_file, records = (root / name for name in
                                               ("input.raw", "stamp.raw", "mask.raw", "placements.bin"))
    result, expected = root / "output.raw", root / "expected.raw"
    source.write_bytes(image)
    stamp_file.write_bytes(stamp)
    mask_file.write_bytes(mask)
    records.write_bytes(b"".join(struct.pack("<iif", *placement) for placement in placements))
    expected.write_bytes(reference(image, width, height, channels, stamp, mask, sw, sh,
                                   placements, 0.6, 255))
    subprocess.run([exe, "weather", "snow_stamp", str(source), str(result), str(width), str(height),
                    str(channels), str(sw), str(sh), str(len(placements)), ".6", "255",
                    str(stamp_file), str(mask_file), str(records)], check=True)
    if result.read_bytes() != expected.read_bytes():
        raise AssertionError("SnowStamp explicit image/mask reference mismatch")

    identity = root / "identity.raw"
    subprocess.run([exe, "snow_stamp", str(source), str(identity), str(width), str(height),
                    str(channels), str(sw), str(sh), "0", "0", "255", "-", "-", "-"], check=True)
    if identity.read_bytes() != image:
        raise AssertionError("zero SnowStamp count/alpha is not identity")

    fill_records, filled = root / "fill.bin", root / "filled.raw"
    fill_records.write_bytes(struct.pack("<iif", 0, 0, 1.0))
    expected_fill = bytearray(image)
    for channel in range(channels):
        expected_fill[channel] = int((image[channel] + 77) * 0.5 + 0.5)
    subprocess.run([exe, "snow_stamp", str(source), str(filled), str(width), str(height),
                    str(channels), "1", "1", "1", ".5", "77", "-", "-",
                    str(fill_records)], check=True)
    if filled.read_bytes() != bytes(expected_fill):
        raise AssertionError("SnowStamp scalar fill fallback mismatch")

print("PASS: SnowStamp exact image/mask, clipping, ordered overlap, fill fallback, and identity reference")
