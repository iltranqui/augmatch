#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(31).integers(0,256,(11,13,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    f=pathlib.Path(td)/"float.raw"
    subprocess.run([exe,"convert","to_float",str(src),str(f),"13","11","3"],check=True)
    got=np.fromfile(f,dtype=np.float32).reshape(image.shape)
    ref=A.ToFloat(max_value=None,p=1)(image=image)["image"]
    np.testing.assert_allclose(got,ref,rtol=0,atol=1e-6)
    out=pathlib.Path(td)/"out.raw"
    subprocess.run([exe,"convert","from_float",str(f),str(out),"13","11","3"],check=True)
    back=np.fromfile(out,dtype=np.uint8).reshape(image.shape)
    ref2=A.FromFloat(max_value=None,p=1)(image=ref)["image"]
    np.testing.assert_array_equal(back,ref2)
print("PASS: convert parity")
