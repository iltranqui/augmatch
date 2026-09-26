#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
np.sctypes={"int":[np.int8,np.int16,np.int32,np.int64],"uint":[np.uint8,np.uint16,np.uint32,np.uint64],"float":[np.float16,np.float32,np.float64],"complex":[np.complex64,np.complex128],"others":[bool,object]}
import imgaug.augmenters as iaa
exe=sys.argv[1]; image=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    ref=A.Transpose(p=1)(image=image)["image"]; dst=pathlib.Path(td)/"transpose.raw"
    subprocess.run([exe,"geometry","transpose",str(src),str(dst),"5","4","3"],check=True)
    np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
    ref=iaa.Rot90(k=1,keep_size=False)(image=image); dst=pathlib.Path(td)/"rot.raw"
    subprocess.run([exe,"geometry","rot90",str(src),str(dst),"5","4","3","1"],check=True)
    np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(ref.shape),ref)
print("PASS: transpose and rotate90 parity exact")
