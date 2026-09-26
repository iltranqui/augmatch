#!/usr/bin/env python3
"""Compare the native JPEG round trip with Pillow, or OpenCV when Pillow is absent."""
import pathlib
import subprocess
import sys
import tempfile

try:
    from PIL import Image
    backend = "Pillow"
except Exception:
    Image = None
    try:
        import cv2
        import numpy as np
        backend = "OpenCV"
    except Exception as exc:
        print(f"jpeg parity skipped: Pillow/OpenCV unavailable ({exc})")
        raise SystemExit(0)

width, height = 32, 24
source = bytes(((x * 7 + channel * 13 + y * 19 + channel * 29 + 3) % 256)
               for y in range(height) for x in range(width) for channel in range(3))
cli = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    raw = directory / "input.raw"
    raw.write_bytes(source)
    for quality, subsampling in ((35, 2), (75, 1), (95, 0)):
        output = directory / f"output-{quality}-{subsampling}.raw"
        subprocess.run([str(cli), "jpeg_compression", str(raw), str(output), str(width), str(height), "3",
                        str(quality), str(subsampling)], check=True)
        actual = output.read_bytes()
        if len(actual) != len(source):
            raise AssertionError(f"native output has {len(actual)} bytes, expected {len(source)}")
        if Image is not None:
            encoded = directory / f"reference-{quality}-{subsampling}.jpg"
            Image.frombytes("RGB", (width, height), source).save(
                encoded, format="JPEG", quality=quality, subsampling=subsampling)
            expected = Image.open(encoded).convert("RGB").tobytes()
        else:
            array = np.frombuffer(source, dtype=np.uint8).reshape(height, width, 3)
            encoded = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            expected = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[:, :, ::-1].tobytes()
        error = max(abs(a - b) for a, b in zip(actual, expected))
        if error > 3:
            raise AssertionError(f"{backend} parity failed for quality={quality}, subsampling={subsampling}: max error {error}")
print(f"jpeg parity passed with {backend}")
