#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.arange(6*8*3,dtype=np.uint8).reshape(6,8,3)
for p in range(4):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"bayer",str(src),str(dst),"8","6",str(p)],check=True)
  ref=np.empty((6,8),np.uint8); maps=[[[0,1],[1,2]],[[2,1],[1,0]],[[1,0],[2,1]],[[1,2],[0,1]]]
  for y in range(6):
   for x in range(8): ref[y,x]=image[y,x,maps[p][y&1][x&1]]
  np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: Bayer sampling parity")
