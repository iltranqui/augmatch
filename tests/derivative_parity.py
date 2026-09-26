#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]; image=np.random.default_rng(115).integers(0,256,(9,10,3),dtype=np.uint8)
refs={
 'sobel_x':cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REPLICATE)),
 'sobel_y':cv2.convertScaleAbs(cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REPLICATE)),
 'scharr_x':cv2.convertScaleAbs(cv2.Scharr(image,cv2.CV_32F,1,0,borderType=cv2.BORDER_REPLICATE)),
 'scharr_y':cv2.convertScaleAbs(cv2.Scharr(image,cv2.CV_32F,0,1,borderType=cv2.BORDER_REPLICATE)),
 'laplacian':cv2.convertScaleAbs(cv2.Laplacian(image,cv2.CV_32F,ksize=1,borderType=cv2.BORDER_REPLICATE)),
 'magnitude':np.clip(np.rint(np.sqrt(cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3,borderType=cv2.BORDER_REPLICATE)**2+cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3,borderType=cv2.BORDER_REPLICATE)**2)),0,255).astype(np.uint8),
}
for op,ref in refs.items():
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
  subprocess.run([exe,'derivative',op,str(src),str(dst),'10','9','3'],check=True)
  got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
  np.testing.assert_allclose(got,ref,atol=1,rtol=0)
for op in ('variance','entropy'):
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
  subprocess.run([exe,'derivative',op,str(src),str(dst),'10','9','3'],check=True)
  expected=np.empty_like(image); padded=np.pad(image,((1,1),(1,1),(0,0)),mode='edge')
  for y in range(image.shape[0]):
   for x in range(image.shape[1]):
    patch=padded[y:y+3,x:x+3]
    if op=='variance': value=np.clip(np.rint(patch.astype(float).var(axis=(0,1))),0,255)
    else:
     value=[]
     for ch in range(3):
      _,counts=np.unique(patch[:,:,ch],return_counts=True); p=counts/counts.sum(); value.append(-np.sum(p*np.log2(p)))
     value=np.clip(np.rint(value),0,255)
    expected[y,x]=value
  got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
  np.testing.assert_allclose(got,expected,atol=1,rtol=0)
print('PASS: derivative parity')
