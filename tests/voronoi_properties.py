#!/usr/bin/env python3
"""Determinism and discrete-target properties for all Voronoi mask APIs."""
import pathlib, subprocess, sys, tempfile

exe=sys.argv[1]; w,h=11,8
source=bytes((i*19+7)%256 for i in range(w*h))
with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); inp=td/'in.raw'; inp.write_bytes(source)
    for op,args in (("voronoi",("5","91")),("uniform_voronoi",("7","91")),("regular_grid_voronoi",("3","2")),("relative_regular_grid_voronoi",(".4",".5"))):
        a=td/(op+'a.raw'); b=td/(op+'b.raw')
        cmd=[exe,op,str(inp),str(a),str(w),str(h),*args]; subprocess.run(cmd,check=True); subprocess.run(cmd[:3]+[str(b),str(w),str(h),*args],check=True)
        got=a.read_bytes(); assert got==b.read_bytes(); assert len(got)==w*h
        # Target semantics: every output byte is an original target label.
        assert set(got).issubset(set(source))
print('PASS: Voronoi deterministic target-mask properties')
