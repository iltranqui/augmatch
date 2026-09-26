#!/usr/bin/env python3
"""Property checks for deterministic RGB photometric contracts."""
import pathlib, subprocess, sys, tempfile

def run(cli, op, data, args):
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d); (p/'in').write_bytes(bytes(data))
        subprocess.run([cli,op,str(p/'in'),str(p/'out'),*map(str,args)],check=True)
        return list((p/'out').read_bytes())

def main():
    x=[0,64,128,255,17,200,90,19]
    y=run(sys.argv[1],'color_clipping',x,[2,1,4,.25,.75])
    assert all(64<=y[i]<=191 for i in (0,1,2,4,5,6))
    assert y[3]==x[3] and y[7]==x[7]
    for op in ('chroma_noise','luma_noise'):
        assert run(sys.argv[1],op,x,[2,1,4,0,9])==x
    assert run(sys.argv[1],'correlated_luma_chroma_noise',x,[2,1,4,0,0,0,9])==x
    return 0
if __name__=='__main__': main()
