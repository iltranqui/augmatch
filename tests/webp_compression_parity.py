#!/usr/bin/env python3
"""Compare native libwebp round trips with Pillow's WebP implementation."""
import pathlib
import subprocess
import sys
import tempfile

try:
    from PIL import Image
except Exception as exc:
    print(f"webp parity skipped: Pillow unavailable ({exc})")
    raise SystemExit(0)

width, height = 32, 24
cases = ((3, "RGB"), (1, "L"))
with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    cli = pathlib.Path(sys.argv[1])
    for channels, mode in cases:
        source = bytes(((x * 7 + channel * 13 + y * 19 + channel * 29 + 3) % 256)
                       for y in range(height) for x in range(width) for channel in range(channels))
        raw = directory / f"input-{channels}.raw"
        raw.write_bytes(source)
        for quality, lossless in ((35, 0), (75, 0), (95, 0), (75, 1)):
            output = directory / f"output-{channels}-{quality}-{lossless}.raw"
            subprocess.run([
                str(cli), "webp_compression", str(raw), str(output), str(width), str(height),
                str(channels), str(quality), str(lossless)
            ], check=True)
            actual = output.read_bytes()
            if len(actual) != len(source):
                raise AssertionError(f"native output has {len(actual)} bytes, expected {len(source)}")
            encoded = directory / f"reference-{channels}-{quality}-{lossless}.webp"
            Image.frombytes(mode, (width, height), source).save(
                encoded, format="WEBP", quality=quality, lossless=bool(lossless), method=4)
            expected = Image.open(encoded).convert(mode).tobytes()
            error = max(abs(a - b) for a, b in zip(actual, expected))
            limit = 0 if lossless else 3
            if error > limit:
                raise AssertionError(
                    f"Pillow parity failed for channels={channels}, quality={quality}, "
                    f"lossless={lossless}: max error {error} > {limit}")
print("webp parity passed with Pillow")
