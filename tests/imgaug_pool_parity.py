#!/usr/bin/env python3
"""Exact uint8 parity checks against imgaug 0.4.0 pooling augmenters."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

# imgaug 0.4.0 predates NumPy 2; provide its removed sctypes compatibility table.
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64],
                  "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                  "float": [np.float16, np.float32, np.float64],
                  "complex": [np.complex64, np.complex128],
                  "others": [bool, object]}
import imgaug.augmenters as iaa


EXECUTABLE = sys.argv[1]
TRANSFORMS = [
    ("AveragePooling", iaa.AveragePooling, 0),
    ("MaxPooling", iaa.MaxPooling, 1),
    ("MinPooling", iaa.MinPooling, 2),
    ("MedianPooling", iaa.MedianPooling, 3),
]


def run_case(image, name, augmenter, mode, kernel, keep_size, raw_in, raw_out):
    image.tofile(raw_in)
    aug = augmenter(kernel_size=kernel, keep_size=keep_size)
    expected = aug(image=image)
    subprocess.run(
        [
            EXECUTABLE,
            "pool",
            str(raw_in),
            str(raw_out),
            str(image.shape[1]),
            str(image.shape[0]),
            str(image.shape[2]),
            str(kernel[0]),
            str(kernel[1]),
            str(mode),
            "1" if keep_size else "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    shape = image.shape if keep_size else (image.shape[0] // kernel[0], image.shape[1] // kernel[1], image.shape[2])
    actual = np.fromfile(raw_out, dtype=np.uint8).reshape(shape)
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
        raise AssertionError(f"{name}, kernel={kernel}, keep_size={keep_size}: {exc}") from exc


def main():
    rng = np.random.default_rng(73129)
    image = rng.integers(0, 256, (24, 30, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_in = pathlib.Path(temp_dir) / "input.raw"
        raw_out = pathlib.Path(temp_dir) / "output.raw"
        cases = 0
        for name, augmenter, mode in TRANSFORMS:
            for kernel in ((2, 2), (3, 3)):
                for keep_size in (False, True):
                    run_case(image, name, augmenter, mode, kernel, keep_size, raw_in, raw_out)
                    cases += 1
    print(f"PASS: {cases} exact imgaug pooling comparisons")


if __name__ == "__main__":
    main()
