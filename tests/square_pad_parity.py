#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.arange(5*7*3,dtype=np.uint8).reshape(5,7,3); side=7
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'size','square_pad',str(src),str(dst),'7','5','3',str(side),str(side),'9','0'],check=True)
 ref=np.full((side,side,3),9,dtype=np.uint8); ref[1:6,:,:]=image
 np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print('PASS: square symmetric pad parity')
