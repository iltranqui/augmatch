#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(90).integers(0,256,(7,9,3),dtype=np.uint8)
for op,gain in [("exposure",1.25),("analog",0.75),("digital",1.5)]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"signal",op,str(src),str(dst),"9","7","3",str(gain)],check=True)
  ref=np.clip(np.rint(image.astype(np.float32)*gain),0,255).astype(np.uint8)
  np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(image.shape),ref,atol=1,rtol=0)
print("PASS: signal gain parity")
