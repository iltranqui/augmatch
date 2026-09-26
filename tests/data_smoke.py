#!/usr/bin/env python3
"""Run a native resize smoke test on repository images."""
import pathlib, subprocess, sys, tempfile
import cv2
import numpy as np
import albumentations as A
exe=sys.argv[1]; images=sorted(pathlib.Path("data").glob("*.jpg"))
if not images: raise SystemExit("no data/*.jpg images found")
with tempfile.TemporaryDirectory() as td:
    for source in images:
        image=cv2.imread(str(source),cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f"failed to read {source}")
        h,w,c=image.shape; ow=max(1,w//3); oh=max(1,h//3)
        raw=pathlib.Path(td)/(source.stem+".raw"); out=pathlib.Path(td)/(source.stem+".out.raw"); image.tofile(raw)
        ref=A.Resize(height=oh,width=ow,interpolation=cv2.INTER_LINEAR,p=1)(image=image)["image"]
        subprocess.run([exe,"resize",str(raw),str(out),str(w),str(h),str(c),str(ow),str(oh),"1"],check=True)
        got=np.fromfile(out,dtype=np.uint8).reshape(ref.shape)
        np.testing.assert_allclose(got,ref,rtol=0,atol=1)
print(f"PASS: repository image smoke parity ({len(images)} images)")
