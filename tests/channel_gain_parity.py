#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(110).integers(0,256,(7,9,3),dtype=np.uint8); gains=(1.1,.8,1.3)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 subprocess.run([exe,"channel_gain",str(src),str(dst),"9","7","3",*map(str,gains)],check=True)
 ref=np.clip(np.rint(image.astype(float)*np.array(gains)),0,255).astype(np.uint8)
 np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(image.shape),ref,atol=1,rtol=0)
print("PASS: channel gain parity")
