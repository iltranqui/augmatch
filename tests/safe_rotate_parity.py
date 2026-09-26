#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]; image=np.random.default_rng(117).integers(0,256,(9,13,3),dtype=np.uint8); angle=27.0
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'safe_rotate',str(src),str(dst),'13','9','3',str(angle),'0'],check=True)
 h,w=image.shape[:2]; a=np.deg2rad(angle); co,si=np.cos(a),np.sin(a); fit=min(1.0,min(w/(abs(co)*w+abs(si)*h),h/(abs(si)*w+abs(co)*h))); co*=fit; si*=fit; cx=(w-1)/2;cy=(h-1)/2
 matrix=np.array([[co,si,cx-co*cx-si*cy],[-si,co,cy+si*cx-co*cy]],dtype=np.float32)
 ref=cv2.warpAffine(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
 np.testing.assert_allclose(got,ref,atol=2,rtol=0)
print('PASS: safe rotate parity')
