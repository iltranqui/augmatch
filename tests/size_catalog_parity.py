#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile

def run(exe, op, image, w, h, args, ow, oh):
    with tempfile.TemporaryDirectory() as td:
        src=pathlib.Path(td)/"in.raw"; dst=pathlib.Path(td)/"out.raw"; src.write_bytes(bytes(image))
        subprocess.run([exe,"size",op,str(src),str(dst),str(w),str(h),"1"]+[str(x) for x in args],check=True)
        data=dst.read_bytes(); assert len(data)==ow*oh
        return data

def main(exe):
    w,h=7,5; image=list(range(w*h))
    # Top-left padding leaves the source origin fixed and uses constant fill.
    d=run(exe,"pad_multiples",image,w,h,[4,3,201,0,0],8,6)
    assert d[:7]==bytes(range(7)) and d[7]==201
    # Center crop uses floor((source-target)/2) for odd differences.
    d=run(exe,"center_crop_multiples",image,w,h,[4,3,0,0,1],4,3)
    assert d==bytes([8,9,10,11,15,16,17,18,22,23,24,25])
    assert len(run(exe,"pad_powers",image,w,h,[2,77,0,0],8,8))==64
    assert len(run(exe,"crop_powers",image,w,h,[2,0,0,0],4,4))==16
    assert len(run(exe,"pad_aspect",image,w,h,[2.0,2,55,0,0],10,5))==50
    assert len(run(exe,"center_crop_aspect",image,w,h,[1.0,2,0,0,1],5,5))==25
    d=run(exe,"keep_size_resize",image,w,h,[3,3,0],w,h)
    assert len(d)==len(image)
    print("PASS: size catalog CPU properties")

if __name__ == "__main__": main(sys.argv[1])
