#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]
rng=np.random.default_rng(11); image=rng.integers(0,256,(13,17,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    for name,aug,mode,sigma in [("average",A.Blur(blur_limit=(3,3),p=1),0,1.0),("gaussian",A.GaussianBlur(blur_limit=(3,3),sigma_limit=(1.0,1.0),p=1),1,1.0),("median",A.MedianBlur(blur_limit=(3,3),p=1),2,1.0)]: 
        ref=aug(image=image)["image"]; dst=pathlib.Path(td)/(name+".raw")
        subprocess.run([exe,"blur",str(src),str(dst),"17","13","3","3",str(sigma),str(mode)],check=True)
        got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
        np.testing.assert_allclose(got,ref,rtol=0,atol=3)
print("PASS: blur parity within three uint8 levels")
