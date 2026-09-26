#!/usr/bin/env python3
"""Dependency-free deterministic and range properties for multi-image CLI."""
import pathlib, subprocess, sys, tempfile

def main(exe):
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d); a=p/'a.raw'; b=p/'b.raw'; o=p/'o.raw'; q=p/'q.raw'
        a.write_bytes(bytes([10])*8); b.write_bytes(bytes([110])*8)
        cmd=[exe,'mixup',str(a),str(b),str(o),'4','2','1','.25','123']
        subprocess.run(cmd,check=True); first=o.read_bytes()
        subprocess.run([exe,'mixup',str(a),str(b),str(q),'4','2','1','.25','123'],check=True)
        assert first==q.read_bytes() and set(first)=={35}
        subprocess.run([exe,'cutmix',str(a),str(b),str(o),'4','2','1','1','0','3','2'],check=True)
        assert o.read_bytes()==bytes([10,110,110,10,10,110,110,10])
    print('mixing properties passed')

if __name__ == '__main__': main(sys.argv[1])
