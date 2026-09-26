#!/usr/bin/env python3
"""Distributional checks for stochastic noise; random streams are not bit-identical."""
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.full((128,128,3),128,dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    out=pathlib.Path(td)/"gauss.raw"
    subprocess.run([exe,"arithmetic","gaussian_noise",str(src),str(out),"128","128","3","20","123"],check=True)
    got=np.fromfile(out,dtype=np.uint8).reshape(image.shape)
    ref=A.GaussNoise(std_range=(20/255,20/255),mean_range=(0,0),p=1)(image=image)["image"]
    assert abs(float(got.mean())-float(ref.mean())) < 3.0
    assert abs(float(got.std())-float(ref.std())) < 3.0
    out=pathlib.Path(td)/"sp.raw"
    subprocess.run([exe,"arithmetic","salt_pepper",str(src),str(out),"128","128","3","0.2","0.5","123"],check=True)
    got=np.fromfile(out,dtype=np.uint8).reshape(image.shape)
    changed=np.mean(got != image)
    assert abs(float(changed)-0.2) < 0.04
print("PASS: stochastic noise distribution checks")
