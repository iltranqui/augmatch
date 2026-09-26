#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
import albumentations as A
exe=sys.argv[1]
rng=np.random.default_rng(20260924)
h,w,c=19,23,3
image=rng.integers(0,256,(h,w,c),dtype=np.uint8)
x,y,cw,ch=3,4,13,11
box=[5.,6.,17.,16.]
point=[8.,9.]
pipeline=A.Compose([A.Crop(x_min=x,y_min=y,x_max=x+cw,y_max=y+ch,p=1.),A.HorizontalFlip(p=1.),A.VerticalFlip(p=1.)],bbox_params=A.BboxParams(format="pascal_voc",label_fields=["labels"]),keypoint_params=A.KeypointParams(format="xy",remove_invisible=False))
ref=pipeline(image=image,bboxes=[box],labels=[7],keypoints=[point])
numpy_ref=image[y:y+ch,x:x+cw].copy()[::-1, ::-1]
np.testing.assert_array_equal(ref["image"],numpy_ref)
with tempfile.TemporaryDirectory() as td:
    src,dst=pathlib.Path(td)/"in.raw",pathlib.Path(td)/"out.raw"
    image.tofile(src)
    cmd=[exe,str(src),str(dst),str(w),str(h),str(c),str(x),str(y),str(cw),str(ch),"1","1",*map(str,box),*map(str,point)]
    proc=subprocess.run(cmd,check=True,text=True,capture_output=True)
    got=np.fromfile(dst,dtype=np.uint8).reshape(ch,cw,c)
    np.testing.assert_array_equal(got,ref["image"])
    values=np.fromstring(proc.stdout,sep=" ")
    # Albumentations uses edge coordinates for boxes and pixel-center coordinates for keypoints.
    np.testing.assert_allclose(values[:4],np.asarray(ref["bboxes"][0][:4]),rtol=0,atol=1e-6)
    np.testing.assert_allclose(values[4:],np.asarray(ref["keypoints"][0][:2]),rtol=0,atol=1e-6)
print("PASS: image pixels are bit-exact with Albumentations and PyTorch; boxes/keypoints match.")
