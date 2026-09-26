#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import cv2
import numpy as np
exe=sys.argv[1]; image=np.arange(5*7*3,dtype=np.uint8).reshape(5,7,3)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    ref=A.CenterCrop(height=3,width=5,p=1)(image=image)["image"]; dst=pathlib.Path(td)/"crop.raw"
    subprocess.run([exe,"size","center_crop",str(src),str(dst),"7","5","3","5","3"],check=True)
    np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
    ref=A.PadIfNeeded(min_height=7,min_width=9,border_mode=cv2.BORDER_CONSTANT,fill=7,position="center",p=1)(image=image)["image"]; dst=pathlib.Path(td)/"pad.raw"
    subprocess.run([exe,"size","pad",str(src),str(dst),"7","5","3","9","7","7","0"],check=True)
    np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: crop and pad parity exact")
