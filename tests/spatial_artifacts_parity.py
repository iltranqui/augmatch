#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
a=.25
for op in ("cross","leak"):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"signal",op,str(src),str(dst),"9","7","3",str(a)],check=True)
  ref=np.empty_like(image)
  for y in range(7):
   for x in range(9):
    for c in range(3):
     v=float(image[y,x,c]); n=float(image[y,max(0,x-1),c]) if op=="leak" else .25*(int(image[y,max(0,x-1),c])+int(image[y,min(8,x+1),c])+int(image[max(0,y-1),x,c])+int(image[min(6,y+1),x,c])); ref[y,x,c]=np.rint((1-a)*v+a*n)
  np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(image.shape),ref,atol=1,rtol=0)
print("PASS: spatial artifact parity")
