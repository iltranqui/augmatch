#!/usr/bin/env python3
"""Compare the deterministic Augmatch CLAHE adapter with OpenCV CLAHE."""
import pathlib
import subprocess
import sys
import tempfile

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"SKIP: CLAHE parity needs numpy and cv2 ({exc})")
    raise SystemExit(0)


def main() -> int:
    cli = pathlib.Path(sys.argv[1])
    rng = np.random.default_rng(20250314)
    height, width, channels = 37, 53, 3
    image = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    # Non-divisible dimensions exercise OpenCV's reflected right/bottom padding.
    clip_limit, tiles_x, tiles_y = 3.5, 7, 5
    expected = np.stack(
        [cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles_x, tiles_y)).apply(image[:, :, channel])
         for channel in range(channels)], axis=2)
    with tempfile.TemporaryDirectory() as directory:
        directory = pathlib.Path(directory)
        source = directory / "input.raw"
        target = directory / "output.raw"
        source.write_bytes(image.tobytes())
        subprocess.run(
            [str(cli), "tone", "clahe", str(source), str(target), str(width), str(height),
             str(channels), str(clip_limit), str(tiles_x), str(tiles_y)],
            check=True,
        )
        actual = np.fromfile(target, dtype=np.uint8).reshape(image.shape)
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    maximum = int(difference.max())
    if maximum > 1:
        raise AssertionError(f"CLAHE differs from OpenCV by {maximum}; mean={difference.mean():.4f}")
    print(f"CLAHE parity passed: max difference {maximum}, mean difference {difference.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
