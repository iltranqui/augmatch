#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(111).integers(0,256,(7,9,3),dtype=np.uint8); ow,oh=13,10
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'scale',str(src),str(dst),'9','7','3',str(ow),str(oh),'1.4'],check=True)
 ref=cv2.resize(image,(ow,oh),interpolation=cv2.INTER_LINEAR)
 got=np.fromfile(dst,dtype=np.uint8).reshape((oh,ow,3))
 np.testing.assert_allclose(got,ref,atol=1,rtol=0)
print('PASS: random scale fixed-factor parity')
