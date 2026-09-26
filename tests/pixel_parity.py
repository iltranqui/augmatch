#!/usr/bin/env python3
"""Parity checks for deterministic pixel transforms."""
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]
rng=np.random.default_rng(7)
image=rng.integers(0,256,(9,11,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; src.write_bytes(image.tobytes())
    cases=[("gamma",A.RandomGamma(gamma_limit=(100,100),p=1),["1.0"]),
           ("brightness_contrast",A.RandomBrightnessContrast(brightness_limit=(0.1,0.1),contrast_limit=(0.0,0.0),p=1),["0.1","1.0"]),
           ("grayscale",A.ToGray(p=1),[])]
    for op,aug,params in cases:
        out=pathlib.Path(td)/(op+".raw")
        ref=aug(image=image)["image"]
        subprocess.run([exe,"pixel",op,str(src),str(out),"11","9","3",*params],check=True)
        got=np.fromfile(out,dtype=np.uint8).reshape(ref.shape)
        np.testing.assert_allclose(got,ref,rtol=0,atol=2)
print("PASS: pixel parity within two uint8 levels")
