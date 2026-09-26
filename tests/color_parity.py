#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    ref=A.RGBShift(r_shift_limit=(10,10),g_shift_limit=(-5,-5),b_shift_limit=(3,3),p=1)(image=image)["image"]; dst=pathlib.Path(td)/"shift.raw"
    subprocess.run([exe,"color","rgb_shift",str(src),str(dst),"7","6","3","10","-5","3"],check=True)
    np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
    dst=pathlib.Path(td)/"shuffle.raw"; subprocess.run([exe,"color","shuffle",str(src),str(dst),"7","6","3","2","0","1"],check=True)
    got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape); np.testing.assert_array_equal(got,image[..., [2,0,1]])
print("PASS: RGB shift and channel shuffle parity")
