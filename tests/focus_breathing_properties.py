import subprocess, sys, tempfile, math
from pathlib import Path

def reference(image, w, h, focus, breathing, radial, cx=.5, cy=.5, fill=0):
    out=[0]*(w*h); ox=cx*(w-1); oy=cy*(h-1); rx=max(1.,(w-1)*.5); ry=max(1.,(h-1)*.5)
    for y in range(h):
        for x in range(w):
            r2=((x-ox)/rx)**2+((y-oy)/ry)**2; scale=1+focus*(breathing+radial*r2); sx=ox+(x-ox)/scale; sy=oy+(y-oy)/scale
            if not math.isfinite(scale) or scale<=0 or not math.isfinite(sx+sy): out[y*w+x]=fill; continue
            x0=math.floor(sx); y0=math.floor(sy); ax=sx-x0; ay=sy-y0
            def at(px,py): return image[py*w+px] if 0<=px<w and 0<=py<h else fill
            value=(1-ay)*((1-ax)*at(x0,y0)+ax*at(x0+1,y0))+ay*((1-ax)*at(x0,y0+1)+ax*at(x0+1,y0+1))
            out[y*w+x]=max(0,min(255,math.floor(value+.5)))
    return bytes(out)

def main():
    exe=sys.argv[1]; w,h,c=11,8,1; image=bytes((17*x+31*y+5)%256 for y in range(h) for x in range(w))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); src=p/'in.raw'; dst=p/'out.raw'; src.write_bytes(image)
        subprocess.run([exe,'focus_breathing',*map(str,[src,dst,w,h,c,'1.25','.22','.08','.5','.5'])],check=True)
        assert dst.read_bytes()==reference(image,w,h,1.25,.22,.08)
        constant=p/'constant.raw'; constant.write_bytes(bytes([77])*(w*h))
        subprocess.run([exe,'focus_breathing',*map(str,[constant,dst,w,h,c,'1','.2','.1','.5','.5'])],check=True); assert dst.read_bytes()==constant.read_bytes()
    print('PASS: focus breathing properties')
if __name__=='__main__': main()
