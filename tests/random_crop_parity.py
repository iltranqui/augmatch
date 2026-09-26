#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(116).integers(0,256,(11,13,3),dtype=np.uint8); ow,oh,left,top=7,6,4,3
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
 subprocess.run([exe,'crop',str(src),str(dst),'13','11','3',str(ow),str(oh),str(left),str(top)],check=True)
 np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape((oh,ow,3)),image[top:top+oh,left:left+ow])
print('PASS: random crop fixed-offset parity')
