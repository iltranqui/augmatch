#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile
exe=sys.argv[1]
# RGB affine is exact and checks CLI argument ordering, alpha preservation, and clipping.
image=bytes([10,20,30,77, 200,100,50,88, 255,0,0,99])
with tempfile.TemporaryDirectory() as td:
    src=pathlib.Path(td)/"in.raw";src.write_bytes(image)
    dst=pathlib.Path(td)/"rgb.raw"
    subprocess.run([exe,"color","with_colorspace",str(src),str(dst),"3","1","4","rgb","2","2","2","1","2","3"],check=True)
    assert dst.read_bytes()==bytes([21,42,63,77,255,202,103,88,255,2,3,99])
    dst=pathlib.Path(td)/"channels.raw"
    subprocess.run([exe,"color","with_brightness_channels",str(src),str(dst),"3","1","4","rgb","1","-1","-1","2","5"],check=True)
    assert dst.read_bytes()==bytes([10,45,30,77,200,205,50,88,255,5,0,99])
    dst=pathlib.Path(td)/"hsv.raw"
    subprocess.run([exe,"color","with_brightness_channels",str(src),str(dst),"3","1","4","hsv","2","-1","-1","0.5","10"],check=True)
    got=dst.read_bytes()
    # Primary red/blue pixels retain hue/saturation while V follows round(V*.5+10).
    assert got[3]==77 and got[7]==88 and got[8:11]==bytes([138,0,0]) and got[11]==99
print("PASS: WithColorspace/WithBrightnessChannels CLI parity")
