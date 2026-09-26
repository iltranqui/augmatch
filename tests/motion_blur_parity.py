#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(80).integers(0,256,(13,15,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 ref=A.MotionBlur(blur_limit=(3,3),angle_range=(0,0),direction_range=(0,0),allow_shifted=False,p=1)(image=image)["image"]
 subprocess.run([exe,"motion_blur",str(src),str(dst),"15","13","3","3","0","0"],check=True)
 np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref,atol=2,rtol=0)
print("PASS: motion blur parity")
