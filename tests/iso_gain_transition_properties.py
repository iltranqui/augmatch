#!/usr/bin/env python3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

def write(path, values):
    path.write_bytes(struct.pack("<%df" % len(values), *values))
def read(path, count):
    return struct.unpack("<%df" % count, path.read_bytes())

def main():
    exe=Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        d=Path(td); knots=d/"knots"; inp=d/"input"; a=d/"a"; b=d/"b"
        write(knots,[100,1,1,1,1,0,1,400,4,1,1,1,0,1]);write(inp,[.1,.25,.5,.9])
        args=[str(exe),"gain_switch_transition",str(inp),str(a),"4","1","1","250",str(knots),"2","1","4","1","2","0",".5",".1","0","19"]
        subprocess.check_call(args); subprocess.check_call(args[:3]+[str(b)]+args[4:])
        x=read(a,4); y=read(b,4)
        if x!=y or any(not (0.0<=v<=1.0) for v in x): raise AssertionError("non-deterministic or unclipped transition")
        if not all(x[i] <= x[i+1] for i in range(3)): raise AssertionError("transition is not monotone")
    return 0
if __name__ == "__main__":
    main()
