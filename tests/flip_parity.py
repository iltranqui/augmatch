#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3)
for axis,aug in [(0,A.VerticalFlip(p=1)),(1,A.HorizontalFlip(p=1))]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"; ref=aug(image=image)["image"]
  subprocess.run([exe,"flip",str(src),str(dst),"10","8","3",str(axis)],check=True)
  np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: generic flip parity")
