#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import numpy as np
exe=sys.argv[1]; image=np.random.default_rng(91).integers(0,256,(7,9,3),dtype=np.uint8)
for op,level in [("saturation",180),("black",-12),("adc",16),("bits",4)]:
 with tempfile.TemporaryDirectory() as td:
  src=pathlib.Path(td)/"in.raw"; image.tofile(src); dst=pathlib.Path(td)/"out.raw"
  subprocess.run([exe,"signal",op,str(src),str(dst),"9","7","3",str(level)],check=True)
  if op=="saturation": ref=np.minimum(image,level)
  elif op=="black": ref=np.clip(image.astype(int)+level,0,255).astype(np.uint8)
  elif op=="adc": ref=np.rint(np.rint(image.astype(float)*(level-1)/255)*(255/(level-1))).astype(np.uint8)
  else: ref=(image.astype(np.uint16) & (255 << (8-level))).astype(np.uint8)
  np.testing.assert_array_equal(np.fromfile(dst,dtype=np.uint8).reshape(image.shape),ref)
print("PASS: signal level parity")
