#!/usr/bin/env python3
import struct, subprocess, sys, tempfile
from pathlib import Path

def floats(path): return struct.unpack("<%df"%(len(path.read_bytes())//4), path.read_bytes())
def main():
    exe=Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as t:
        d=Path(t); inp=d/'in'; lut=d/'lut'; a=d/'a'; b=d/'b'; maps=d/'maps'; ca=d/'ca'; cb=d/'cb'
        inp.write_bytes(struct.pack('<4f',.1,.2,.3,.4))
        lut.write_bytes(struct.pack('<20f',100,1,1,1,1,0,1,100,0,0,400,2,1,1,1,0,1,100,0,0))
        args=[str(exe),'camera_profile_lookup',str(inp),str(a),'4','1','1','250',str(lut),'2','9']
        subprocess.check_call(args); subprocess.check_call(args[:3]+[str(b)]+args[4:])
        x=floats(a); y=floats(b)
        if x!=y or any(not (0<=v<=1) for v in x): raise AssertionError('camera LUT operation is not deterministic/clipped')
        maps.write_bytes(struct.pack('<12f', *([0.0,.01,.02,.03]+[0.0]*4+[1.0]*4)))
        args=[str(exe),'calibration_noise',str(inp),str(ca),'4','1','1','0','0','0','7',str(maps)]
        subprocess.check_call(args); subprocess.check_call(args[:3]+[str(cb)]+args[4:])
        if floats(ca)!=floats(cb): raise AssertionError('calibration map operation is not deterministic')
    print('PASS: camera profile LUT and calibration properties')
if __name__=='__main__': main()
