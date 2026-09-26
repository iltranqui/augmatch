#!/usr/bin/env python3
"""Reference and property checks for the libjpeg-backed catalog entries."""
import pathlib
import subprocess
import sys
import tempfile

try:
    from PIL import Image
except Exception as exc:
    print(f"jpeg catalog reference skipped: Pillow unavailable ({exc})")
    raise SystemExit(0)

w, h, c = 40, 32, 3
source = bytes(((x * 13 + y * 5 + channel * 31) & 255)
               for y in range(h) for x in range(w) for channel in range(c))
cli = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    raw = td / "input.raw"
    raw.write_bytes(source)
    # Pillow is the reference for the quality and sampling controls.
    for quality, sampling in ((25, 2), (55, 1), (90, 0)):
        actual_path = td / f"actual-{quality}-{sampling}.raw"
        subprocess.run([str(cli), "jpeg_quality_variation", str(raw), str(actual_path),
                        str(w), str(h), str(c), str(quality), str(sampling)], check=True)
        actual = actual_path.read_bytes()
        encoded = td / f"reference-{quality}-{sampling}.jpg"
        Image.frombytes("RGB", (w, h), source).save(encoded, format="JPEG",
                                                     quality=quality, subsampling=sampling)
        expected = Image.open(encoded).convert("RGB").tobytes()
        if actual != expected:
            raise AssertionError(f"quality/sampling mismatch at quality={quality}, sampling={sampling}")
    outputs = []
    for operation, args in (
        ("jpeg_quantization_table_variation", (55, 2.0, 1.0, 2)),
        ("jpeg_ringing", (35, 0.5, 2)),
        ("jpeg_blocking", (35, 0.5, 2)),
        ("jpeg_mosquito_noise", (35, 0.5, 2)),
        ("jpeg_restart_marker_damage", (35, 1.0, 2, 4)),
        ("jpeg_progressive_decoding", (35, 1.0, 2, 1)),
    ): 
        out = td / f"{operation}.raw"
        if operation == "jpeg_quantization_table_variation":
            command = [str(cli), operation, str(raw), str(out), str(w), str(h), str(c),
                       str(args[0]), str(args[1]), str(args[2]), str(args[3])]
        elif operation == "jpeg_restart_marker_damage":
            command = [str(cli), operation, str(raw), str(out), str(w), str(h), str(c),
                       str(args[0]), str(args[2]), str(args[1]), str(args[3])]
        elif operation == "jpeg_progressive_decoding":
            command = [str(cli), operation, str(raw), str(out), str(w), str(h), str(c),
                       str(args[0]), str(args[2]), str(args[1]), str(args[3])]
        else:
            command = [str(cli), operation, str(raw), str(out), str(w), str(h), str(c),
                       str(args[0]), str(args[2]), str(args[1])]
        subprocess.run(command, check=True)
        data = out.read_bytes()
        if len(data) != len(source) or not all(0 <= value <= 255 for value in data):
            raise AssertionError(f"{operation} violated uint8 shape/range property")
        outputs.append(data)
    if len(set(outputs)) != len(outputs):
        raise AssertionError("JPEG catalog artifact operations unexpectedly alias one another")
print("jpeg catalog reference/property checks passed")
