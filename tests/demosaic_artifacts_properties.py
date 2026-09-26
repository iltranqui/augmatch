#!/usr/bin/env python3
"""Reference and deterministic properties for catalog demosaicing artifacts."""
import pathlib, subprocess, sys, tempfile
exe=sys.argv[1]; w,h=11,9
raw=bytes((i*29+(i//w)*17)&255 for i in range(w*h))
with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); src=td/'in.raw'; src.write_bytes(raw)
    base=td/'base.raw'; subprocess.run([exe,'demosaic_edge_aware',str(src),str(base),str(w),str(h),'0'],check=True); reference=base.read_bytes()
    commands=[('demosaic_directional',['0','0.0']),('demosaic_zipper',['0.0']),('demosaic_aliasing',['3','0.25','0.0']),('demosaic_ringing',['0.0']),('demosaic_noise_amplification',['0.0','2.0','123'])]
    for op,args in commands:
        a=td/(op+'a.raw'); b=td/(op+'b.raw')
        subprocess.run([exe,op,str(src),str(a),str(w),str(h),'0',*args],check=True)
        subprocess.run([exe,op,str(src),str(b),str(w),str(h),'0',*args],check=True)
        assert a.read_bytes()==b.read_bytes(), op
        assert a.read_bytes()==reference, op+' zero-strength reference mismatch'
        assert len(a.read_bytes())==w*h*3
print('PASS: demosaicing artifact deterministic/reference properties')
