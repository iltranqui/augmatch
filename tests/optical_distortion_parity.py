#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]; image=np.random.default_rng(119).integers(0,256,(9,12,3),dtype=np.uint8); k1,k2,p1,p2=.08,-.015,.01,-.006
h,w=image.shape[:2]; yy,xx=np.mgrid[0:h,0:w].astype(np.float32); cx=(w-1)*.5;cy=(h-1)*.5;fx=max(1.,w*.5);fy=max(1.,h*.5);xn=(xx-cx)/fx;yn=(yy-cy)/fy;r2=xn*xn+yn*yn;rad=1+k1*r2+k2*r2*r2
mapx=(xn*rad+2*p1*xn*yn+p2*(r2+2*xn*xn))*fx+cx; mapy=(yn*rad+p1*(r2+2*yn*yn)+2*p2*xn*yn)*fy+cy
ref=cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'optical_distortion',str(src),str(dst),str(w),str(h),'3',str(k1),str(k2),str(p1),str(p2),'0'],check=True)
 got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
 np.testing.assert_allclose(got,ref,atol=2,rtol=0)
print('PASS: optical distortion parity')
