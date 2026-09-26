#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(40).integers(0,256,(7,13,3),dtype=np.uint8)
for op,aug in [("longest",A.LongestMaxSize(max_size=9,p=1)),("smallest",A.SmallestMaxSize(max_size=9,p=1))]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"; ref=aug(image=image)["image"]
  h,w,c=ref.shape
  subprocess.run([exe,"maxsize",op,str(src),str(dst),"13","7","3","9",str(w),str(h)],check=True)
  np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref,atol=2,rtol=0)
print("PASS: max-size parity")
