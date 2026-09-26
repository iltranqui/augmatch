#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]; image=np.random.default_rng(112).integers(0,256,(8,9,3),dtype=np.uint8)
for axis in (0,1):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src); factor=1.25
  subprocess.run([exe,'axis_scale',str(src),str(dst),'9','8','3',str(axis),str(factor)],check=True)
  t=(1-factor)*((image.shape[1 if axis==0 else 0]-1)*.5)
  matrix=np.array([[factor,0,t],[0,1,0]]) if axis==0 else np.array([[1,0,0],[0,factor,t]])
  ref=cv2.warpAffine(image,matrix,(9,8),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
  got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
  np.testing.assert_allclose(got,ref,atol=2,rtol=0)
print('PASS: axis scale parity')
