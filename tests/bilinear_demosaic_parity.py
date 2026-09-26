#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; mosaic=np.arange(6*8,dtype=np.uint8).reshape(6,8)
for p in range(4):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; mosaic.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"demosaic_bilinear",str(src),str(dst),"8","6",str(p)],check=True)
  maps=[[[0,1],[1,2]],[[2,1],[1,0]],[[1,0],[2,1]],[[1,2],[0,1]]]; ref=np.empty((6,8,3),np.uint8)
  for y in range(6):
   for x in range(8):
    for c in range(3):
     vals=[int(mosaic[max(0,min(5,y+dy)),max(0,min(7,x+dx))]) for dy in (-1,0,1) for dx in (-1,0,1) if maps[p][max(0,min(5,y+dy))&1][max(0,min(7,x+dx))&1]==c]
     ref[y,x,c]=(sum(vals)+len(vals)//2)//len(vals)
  np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: bilinear demosaicing parity")
