#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import cv2
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(61).integers(0,256,(13,15,3),dtype=np.uint8)
# Fixed ranges make the reference deterministic.
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 # Albumentations documents ShiftScaleRotate as a special case of Affine;
 # use its deterministic equivalent for exact reference parity.
 ref=A.Affine(scale=1.2,translate_px={"x":2,"y":1},rotate=15,interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,fill=0,p=1)(image=image)["image"]
 subprocess.run([exe,"shift_scale_rotate",str(src),str(dst),"15","13","3",str(2/15),str(1/13),"1.2","15","0"],check=True)
 np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref,atol=2,rtol=0)
print("PASS: shift-scale-rotate parity")
