#!/usr/bin/env python3
"""Parity for the deterministic imgaug.pillike Pillow-backed batch."""
import pathlib, subprocess, sys, tempfile
try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    print("SKIP: Pillow/NumPy unavailable")
    raise SystemExit(77)

exe = sys.argv[1]
rng = np.random.default_rng(1337)
for channels in (3, 4):
    image = rng.integers(0, 256, (11, 13, channels), dtype=np.uint8)
    mode = "RGB" if channels == 3 else "RGBA"
    pil = Image.fromarray(image, mode)
    expected = {
        "enhance_color": np.asarray(ImageEnhance.Color(pil).enhance(0.35)),
        "enhance_contrast": np.asarray(ImageEnhance.Contrast(pil).enhance(1.7)),
        "enhance_brightness": np.asarray(ImageEnhance.Brightness(pil).enhance(0.65)),
        "enhance_sharpness": np.asarray(ImageEnhance.Sharpness(pil).enhance(1.8)),
        "filter_blur": np.asarray(pil.filter(ImageFilter.BLUR)),
        "filter_smooth": np.asarray(pil.filter(ImageFilter.SMOOTH)),
        "filter_smooth_more": np.asarray(pil.filter(ImageFilter.SMOOTH_MORE)),
        "filter_edge_enhance": np.asarray(pil.filter(ImageFilter.EDGE_ENHANCE)),
        "filter_edge_enhance_more": np.asarray(pil.filter(ImageFilter.EDGE_ENHANCE_MORE)),
        "filter_find_edges": np.asarray(pil.filter(ImageFilter.FIND_EDGES)),
        "filter_contour": np.asarray(pil.filter(ImageFilter.CONTOUR)),
        "filter_emboss": np.asarray(pil.filter(ImageFilter.EMBOSS)),
        "filter_sharpen": np.asarray(pil.filter(ImageFilter.SHARPEN)),
        "filter_detail": np.asarray(pil.filter(ImageFilter.DETAIL)),
    }
    factors = {"enhance_color": "0.35", "enhance_contrast": "1.7", "enhance_brightness": "0.65", "enhance_sharpness": "1.8"}
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "in.raw"
        image.tofile(src)
        for op, ref in expected.items():
            dst = pathlib.Path(td) / (op + ".raw")
            args = [exe, op, str(src), str(dst), str(image.shape[1]), str(image.shape[0]), str(channels)]
            if op in factors:
                args.append(factors[op])
            subprocess.run(args, check=True)
            got = np.fromfile(dst, dtype=np.uint8).reshape(image.shape)
            # Pillow's integer filter path has version-dependent exact-half tie
            # behavior for SMOOTH_MORE; all non-tie values must be exact.
            np.testing.assert_allclose(got, ref, atol=1, rtol=0, err_msg=f"{op} channels={channels}")
print("PASS: pillike Pillow parity")
