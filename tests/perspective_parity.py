#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]; image=np.random.default_rng(118).integers(0,256,(9,12,3),dtype=np.uint8); h=(1.,.02,-.8,-.01,1.,.6,.0004,-.0003,1.)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'perspective',str(src),str(dst),'12','9','3',*map(str,h)],check=True)
 ref=cv2.warpPerspective(image,np.array(h,dtype=np.float32).reshape(3,3),(12,9),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
 np.testing.assert_allclose(got,ref,atol=2,rtol=0)
 dst2=pathlib.Path(td)/'out-transform.raw'
 subprocess.run([exe,'perspective_transform',str(src),str(dst2),'12','9','3',*map(str,h)],check=True)
 np.testing.assert_array_equal(np.fromfile(dst2,dtype=np.uint8),got.ravel())

 # A valid homography can still project an output pixel onto its point at
 # infinity.  Both an exact and a near-zero denominator must use the fill
 # value instead of reaching the coordinate conversion/sampler with invalid
 # coordinates.  The CLI configures the documented default fill value (0).
 singular_cases = [
     ("zero denominator", (1.,0.,0.,0.,1.,0.,1.,0.,-1.)),
     ("near-zero denominator", (1.,0.,0.,0.,1.,0.,1.,0.,-0.99999994)),
 ]
 with tempfile.TemporaryDirectory() as td:
     src=pathlib.Path(td)/'singular-in.raw'; image.tofile(src)
     for name, singular_h in singular_cases:
         dst=pathlib.Path(td)/(name.replace(' ', '-')+'-out.raw')
         subprocess.run([exe,'perspective',str(src),str(dst),'12','9','3',*map(str,singular_h)],check=True)
         got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
         np.testing.assert_array_equal(got[:,1,:],0)
print('PASS: perspective parity and invalid-coordinate fill handling')
