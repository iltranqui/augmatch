#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(100).integers(0,256,(13,15,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 ref=A.Defocus(radius=(2,2),alias_blur=(0,0),p=1)(image=image)["image"]
 subprocess.run([exe,"defocus",str(src),str(dst),"15","13","3","2","0"],check=True)
 np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref,atol=3,rtol=0)
print("PASS: defocus parity")
