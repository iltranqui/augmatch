#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(19).integers(0,256,(17,19,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
    ref=A.HueSaturationValue(hue_shift_limit=(10,10),sat_shift_limit=(-15,-15),val_shift_limit=(7,7),p=1)(image=image)["image"]
    subprocess.run([exe,"color","hsv_shift",str(src),str(dst),"19","17","3","10","-15","7"],check=True)
    got=np.fromfile(dst,dtype=np.uint8).reshape(ref.shape)
    np.testing.assert_allclose(got,ref,rtol=0,atol=8)
print("PASS: HSV shift parity within eight uint8 levels")
