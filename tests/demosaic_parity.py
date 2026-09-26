#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; mosaic=np.arange(6*8,dtype=np.uint8).reshape(6,8)
for p in range(4):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; mosaic.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"demosaic",str(src),str(dst),"8","6",str(p)],check=True)
  maps=[[[0,1],[1,2]],[[2,1],[1,0]],[[1,0],[2,1]],[[1,2],[0,1]]]; ref=np.empty((6,8,3),np.uint8)
  for y in range(6):
   for x in range(8):
    for c in range(3):
     found=None
     for r in range(3):
      for dy in range(-r,r+1):
       for dx in range(-r,r+1):
        yy=max(0,min(5,y+dy));xx=max(0,min(7,x+dx))
        if abs(dx)+abs(dy)==r and maps[p][yy&1][xx&1]==c and found is None: found=mosaic[yy,xx]
      if found is not None: break
     ref[y,x,c]=found
  np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: nearest demosaicing parity")
