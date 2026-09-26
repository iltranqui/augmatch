#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
# Albumentations uses positive values for padding and negative values for cropping.
for px in [(1,2,1,2),(-1,-2,-1,-2),(1,-2,2,-1)]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  ref=A.CropAndPad(px=px,keep_size=False,p=1,border_mode=0,fill=17)(image=image)["image"]
  h,w,c=ref.shape
  subprocess.run([exe,"crop_pad",str(src),str(dst),"9","7","3",*map(str,px),"17"],check=True)
  got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
  np.testing.assert_array_equal(got,ref)
print("PASS: crop-and-pad parity")
