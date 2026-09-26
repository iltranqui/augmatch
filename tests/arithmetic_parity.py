#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
import albumentations as A
np.sctypes={"int":[np.int8,np.int16,np.int32,np.int64],"uint":[np.uint8,np.uint16,np.uint32,np.uint64],"float":[np.float16,np.float32,np.float64],"complex":[np.complex64,np.complex128],"others":[bool,object]}
import imgaug.augmenters as iaa
exe=sys.argv[1]; image=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)*3
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    cases=[("add",iaa.Add(10),["10"]),("multiply",iaa.Multiply(1.5),["1.5"]),("invert",A.InvertImg(p=1),[]),("posterize",A.Posterize(num_bits=4,p=1),["4"]),("solarize",A.Solarize(threshold_range=(0.5,0.5),p=1),["128"])]
    for op,aug,args in cases:
        ref=aug(image=image) if op in ("add","multiply") else aug(image=image)["image"]
        if op in ("add","multiply"): ref=ref
        dst=pathlib.Path(td)/(op+".raw")
        subprocess.run([exe,"arithmetic",op,str(src),str(dst),"5","4","3",*args],check=True)
        got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
        np.testing.assert_array_equal(got,ref)
print("PASS: arithmetic parity exact")
