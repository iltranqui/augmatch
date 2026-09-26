#!/usr/bin/env python3
"""Exact imgaug 0.4.0 convolutional parity through OpenCV filter2D."""
import pathlib
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def apply(image, effect, alpha):
    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1.0
    matrix = (1.0 - alpha) * identity + alpha * effect
    channels = [cv2.filter2D(image[..., ch], cv2.CV_32F, matrix,
                             borderType=cv2.BORDER_DEFAULT)
                for ch in range(image.shape[2])]
    # The native APIs convert the OpenCV float response to uint8 with
    # saturating nearest-integer rounding, matching cvRound away from the
    # integer tie ambiguity in OpenCV's direct uint8 filter path.
    return np.clip(np.floor(np.stack(channels, axis=2) + np.float32(0.5)),
                   0, 255).astype(np.uint8)


def directed_effect(direction):
    degrees = int(direction * 360) % 360
    radians = np.deg2rad(degrees)
    vector = np.array([np.cos(radians - 0.5 * np.pi),
                       np.sin(radians - 0.5 * np.pi)])
    effect = np.zeros((3, 3), dtype=np.float32)
    effect[1, 1] = 1.0
    total = 0.0
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            if x == 0 and y == 0:
                continue
            distance = np.rad2deg(np.arccos(np.clip(np.dot((x, y), vector)
                                                    / np.hypot(x, y), -1, 1)))
            similarity = (1.0 - distance / 180.0) ** 4
            effect[y + 1, x + 1] = -similarity
            total += similarity
    effect *= np.float32(1.0 / total)
    effect[1, 1] = 1.0
    return effect


EXE = sys.argv[1]
rng = np.random.default_rng(20250321)
image = rng.integers(0, 256, (17, 19, 3), dtype=np.uint8)
emboss_strength = 1.25
emboss_effect = np.array([[-1 - emboss_strength, -emboss_strength, 0],
                          [-emboss_strength, 1, emboss_strength],
                          [0, emboss_strength, 1 + emboss_strength]], dtype=np.float32)
edge_effect = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
cases = [
    ("emboss", [0.63, emboss_strength], apply(image, emboss_effect, 0.63)),
    ("edge_detect", [0.71], apply(image, edge_effect, 0.71)),
    ("directed_edge_detect", [0.58, 0.42], apply(image, directed_effect(0.42), 0.58)),
]
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    source, destination = root / "in.raw", root / "out.raw"
    image.tofile(source)
    for operation, parameters, expected in cases:
        subprocess.run([EXE, operation, str(source), str(destination), "19", "17", "3",
                        *(str(x) for x in parameters)], check=True)
        actual = np.fromfile(destination, dtype=np.uint8).reshape(image.shape)
        np.testing.assert_array_equal(actual, expected, err_msg=operation)
print("PASS: exact imgaug convolutional OpenCV parity (Emboss/EdgeDetect/DirectedEdgeDetect)")
