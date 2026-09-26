#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
import cv2
import numpy as np
exe=sys.argv[1]
rng=np.random.default_rng(83); image=rng.integers(0,256,(9,11,4),dtype=np.uint8)
def reference(op, *args):
    rgb=image[:,:,:3]; hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    if op == "multiply_brightness": hsv[:,:,2]=np.clip(np.floor(hsv[:,:,2]*args[0]+.5),0,255)
    elif op == "add_to_brightness": hsv[:,:,2]=np.clip(np.floor(hsv[:,:,2]+args[0]+.5),0,255)
    elif op == "multiply_and_add_to_brightness": hsv[:,:,2]=np.clip(np.floor(hsv[:,:,2]*args[0]+args[1]+.5),0,255)
    elif op == "with_hue_and_saturation": hsv[:,:,0]=args[0]%180; hsv[:,:,1]=np.clip(np.floor(args[1]+.5),0,255)
    elif op == "multiply_hue_and_saturation": hsv[:,:,0]=(np.floor(hsv[:,:,0]*args[0]+.5)%180); hsv[:,:,1]=np.clip(np.floor(hsv[:,:,1]*args[1]+.5),0,255)
    elif op == "multiply_hue": hsv[:,:,0]=np.floor(hsv[:,:,0]*args[0]+.5)%180
    elif op == "multiply_saturation": hsv[:,:,1]=np.clip(np.floor(hsv[:,:,1]*args[0]+.5),0,255)
    elif op == "remove_saturation": hsv[:,:,1]=0
    elif op == "add_to_hue_and_saturation": hsv[:,:,0]=(np.floor(hsv[:,:,0]+args[0]+.5)%180); hsv[:,:,1]=np.clip(np.floor(hsv[:,:,1]+args[1]+.5),0,255)
    elif op == "add_to_hue": hsv[:,:,0]=np.floor(hsv[:,:,0]+args[0]+.5)%180
    elif op == "add_to_saturation": hsv[:,:,1]=np.clip(np.floor(hsv[:,:,1]+args[0]+.5),0,255)
    result=cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2RGB)
    return np.dstack((result,image[:,:,3:]))
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw"; image.tofile(src)
    cases=[("multiply_brightness",[.65]),("add_to_brightness",[19]),("multiply_and_add_to_brightness",[.65,19]),("with_hue_and_saturation",[62,177]),("multiply_hue_and_saturation",[1.3,.7]),("multiply_hue",[1.7]),("multiply_saturation",[.4]),("remove_saturation",[]),("add_to_hue_and_saturation",[-23,31]),("add_to_hue",[47]),("add_to_saturation",[-29])]
    for index,(op,args) in enumerate(cases):
        dst=pathlib.Path(td)/f"out{index}.raw"; subprocess.run([exe,"color",op,str(src),str(dst),"11","9","4",*(str(x) for x in args)],check=True)
        got=np.fromfile(dst,dtype=np.uint8).reshape(image.shape)
        np.testing.assert_allclose(got,reference(op,*args),rtol=0,atol=6)
print("PASS: HSV color batch CPU parity within six uint8 levels")
