#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(81).integers(0,256,(9,11,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 ref=A.MultiplicativeNoise(multiplier=(1.2,1.2),per_channel=False,elementwise=False,p=1)(image=image)["image"]
 subprocess.run([exe,"arithmetic","multiply",str(src),str(dst),"11","9","3","1.2"],check=True)
 np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: multiplicative-noise fixed multiplier parity")
