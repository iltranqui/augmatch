#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import cv2
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(60).integers(0,256,(11,13,3),dtype=np.uint8)
for angle in (0,15,-20):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  ref=A.Rotate(limit=(angle,angle),interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,fill=0,p=1)(image=image)["image"]
  subprocess.run([exe,"rotate",str(src),str(dst),"13","11","3",str(angle),"0"],check=True)
  got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
  np.testing.assert_allclose(got,ref,atol=2,rtol=0)
print("PASS: rotate parity")
