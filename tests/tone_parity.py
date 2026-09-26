#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(21).integers(0,256,(17,19,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    for op,aug in [("sepia",A.ToSepia(p=1)),("autocontrast",A.AutoContrast(method="pil",p=1)),("equalize",A.Equalize(mode="cv",by_channels=True,p=1))]:
        ref=aug(image=image)["image"]; dst=pathlib.Path(td)/(op+".raw")
        subprocess.run([exe,"tone",op,str(src),str(dst),"19","17","3"],check=True)
        got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
        np.testing.assert_allclose(got,ref,rtol=0,atol=2)
print("PASS: tone parity within two uint8 levels")
