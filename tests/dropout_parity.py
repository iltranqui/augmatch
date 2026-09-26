#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import albumentations as A
import numpy as np
exe=sys.argv[1]; image=np.full((64,64,3),127,dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    def run(op,args):
        dst=pathlib.Path(td)/(op+".raw"); subprocess.run([exe,"dropout",op,str(src),str(dst),"64","64","3",*map(str,args)],check=True); return np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
    got=run("pixel",[.25,0,123]); assert abs(float(np.mean(got==0))-.25)<.04
    got=run("channel",[.5,0,123]); assert np.sum(np.all(got==0,axis=(0,1))) in (0,1,2,3)
    got=run("grid",[4,4,.5,0]); ref=A.GridDropout(ratio=.5,random_offset=False,unit_size_range=(4,5),fill=0,p=1)(image=image)["image"]; assert abs(float(np.mean(got==0))-float(np.mean(ref==0)))<.05
    got=run("coarse",[4,8,8,0,123]); assert abs(float(np.mean(got==0))-.0625)<.05
print("PASS: dropout distribution and grid checks")
