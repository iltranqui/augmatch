#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(83).integers(0,256,(8,10,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
 src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
 ref=A.RandomGamma(gamma_limit=(120,120),p=1)(image=image)["image"]
 subprocess.run([exe,"arithmetic","gamma",str(src),str(dst),"10","8","3","1.2"],check=True)
 np.testing.assert_allclose(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref,atol=1,rtol=0)
print("PASS: gamma parity")
