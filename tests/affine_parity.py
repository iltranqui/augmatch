#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import cv2
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(50).integers(0,256,(9,11,3),dtype=np.uint8)
for tr in [(0,0),(2,-1)]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  ref=A.Affine(scale=1,translate_px={"x":tr[0],"y":tr[1]},rotate=0,shear=0,fit_output=False,interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,fill=0,p=1)(image=image)["image"]
  tx,ty=tr
  # The API matrix is the forward input-to-output matrix used by this native wrapper.
  subprocess.run([exe,"affine",str(src),str(dst),"11","9","3","11","9","1","0",str(tx),"0","1",str(ty)],check=True)
  got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
  np.testing.assert_allclose(got,ref,atol=2,rtol=0)
print("PASS: affine parity")
