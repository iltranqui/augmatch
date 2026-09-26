#!/usr/bin/env python3
"""Check both explicit all-channel contrast aliases against OpenCV."""
import pathlib
import subprocess
import sys
import tempfile

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"SKIP: all-channel contrast parity needs numpy and cv2 ({exc})")
    raise SystemExit(0)


def run(cli, op, image, args, directory):
    source = directory / f"{op}.in.raw"
    target = directory / f"{op}.out.raw"
    source.write_bytes(image.tobytes())
    subprocess.run([str(cli), "tone", op, str(source), str(target), *map(str, args)], check=True)
    return np.fromfile(target, dtype=np.uint8).reshape(image.shape)


def main() -> int:
    cli = pathlib.Path(sys.argv[1])
    height, width, channels = 29, 37, 4
    image = np.random.default_rng(20250315).integers(0, 256, (height, width, channels), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        clahe = run(cli, "all_channels_clahe", image, (width, height, channels, 3.5, 5, 4), directory)
        expected_clahe = np.stack(
            [cv2.createCLAHE(clipLimit=3.5, tileGridSize=(5, 4)).apply(image[:, :, channel])
             for channel in range(channels)], axis=2)
        equalized = run(cli, "all_channels_histogram_equalization", image,
                        (width, height, channels), directory)
        expected_equalized = np.stack(
            [cv2.equalizeHist(image[:, :, channel]) for channel in range(channels)], axis=2)
    clahe_difference = np.abs(clahe.astype(np.int16) - expected_clahe.astype(np.int16))
    equalize_difference = np.abs(equalized.astype(np.int16) - expected_equalized.astype(np.int16))
    if clahe_difference.max() > 1 or equalize_difference.max() > 1:
        raise AssertionError(
            f"all-channel contrast mismatch: CLAHE max={clahe_difference.max()}, "
            f"equalize max={equalize_difference.max()}")
    print("all-channel contrast parity passed: max difference "
          f"CLAHE={clahe_difference.max()}, equalize={equalize_difference.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
