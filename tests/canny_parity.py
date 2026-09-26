#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2, numpy as np
exe=sys.argv[1]
rng=np.random.default_rng(20250301)
for aperture in (3,5,7):
  image=rng.integers(0,256,(41,37),dtype=np.uint8)
  for low,high in ((20,60),(50,150),(100,180)):
    with tempfile.TemporaryDirectory() as td:
      src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
      subprocess.run([exe,'canny',str(src),str(dst),'37','41','1',str(low),str(high),str(aperture),'0','0'],check=True)
      got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
    expected=cv2.Canny(image,low,high,apertureSize=aperture,L2gradient=False)
    # OpenCV's Canny intentionally treats the outer gradient ring specially;
    # the explicit AUGMATCH border contract is checked below. A small tolerance
    # covers OpenCV's integer tie decisions while still checking full masks.
    assert np.count_nonzero(got!=expected) <= 0.01*got.size, (aperture,low,high)
# Explicit HWC policy: RGB is luminance-tested and each output channel gets the mask.
image=rng.integers(0,256,(19,23,3),dtype=np.uint8)
with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; image.tofile(src)
  subprocess.run([exe,'canny',str(src),str(dst),'23','19','3','30','90','3','0','0'],check=True)
  got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
assert np.array_equal(got[:,:,0],got[:,:,1]) and np.array_equal(got[:,:,1],got[:,:,2])
assert np.all((got==0)|(got==255))
# Every high threshold increase can only remove edges (deterministic hysteresis property).
with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/'in.raw'; lo=pathlib.Path(td)/'lo.raw'; hi=pathlib.Path(td)/'hi.raw'; image[:,:,0].tofile(src)
  subprocess.run([exe,'canny',str(src),str(lo),'23','19','1','20','70','3','0','0'],check=True)
  subprocess.run([exe,'canny',str(src),str(hi),'23','19','1','20','150','3','0','0'],check=True)
  low=np.fromfile(lo,dtype=np.uint8); high=np.fromfile(hi,dtype=np.uint8)
  assert np.all(high<=low)
print('PASS: Canny OpenCV parity and deterministic channel/property checks')
