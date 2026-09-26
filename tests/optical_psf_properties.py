import subprocess, sys, tempfile
from pathlib import Path

def run(exe, op, args): subprocess.run([exe, op, *map(str,args)], check=True)
def main():
    exe=sys.argv[1]; w,h,c=9,7,1
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); src=p/'in.raw'; out=p/'out.raw'; data=bytes((x+19*y)%256 for y in range(h) for x in range(w)); src.write_bytes(data)
        run(exe,'diffraction_blur',[src,out,w,h,c,0,550,2,50]); assert out.read_bytes()==data
        run(exe,'bokeh_blur',[src,out,w,h,c,0,0]); assert out.read_bytes()==data
        run(exe,'cat_eye_bokeh',[src,out,w,h,c,0,.5,.5,.5]); assert out.read_bytes()==data
        run(exe,'aperture_shape_blur',[src,out,w,h,c,0,6,0,0]); assert out.read_bytes()==data
        constant=p/'constant.raw'; constant.write_bytes(bytes([77])*(w*h)); run(exe,'aperture_shape_blur',[constant,out,w,h,c,4,7,13,.25]); assert out.read_bytes()==constant.read_bytes()
    print('PASS: optical PSF properties')
if __name__=='__main__': main()
