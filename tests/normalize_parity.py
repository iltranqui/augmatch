#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(32).integers(0,256,(9,12,3),dtype=np.uint8)
mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 subprocess.run([exe,"normalize",str(src),str(dst),"12","9","3",*map(str,mean),*map(str,std)],check=True)
 got=np.fromfile(dst,dtype=np.float32).reshape(image.shape)
 ref=A.Normalize(mean=mean,std=std,max_pixel_value=255,p=1)(image=image)["image"]
 np.testing.assert_allclose(got,ref,rtol=1e-6,atol=1e-6)
print("PASS: normalize parity")
