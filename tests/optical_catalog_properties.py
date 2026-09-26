import struct, subprocess, sys, tempfile
from pathlib import Path

def main():
    exe=sys.argv[1]
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); w,h,c=7,5,3; src=p/'in.raw'; src.write_bytes(bytes((i*13)%256 for i in range(w*h*c))); out=p/'out.raw'
        run=lambda op,args: subprocess.run([exe,op,*map(str,args)],check=True)
        run('optical_defocus',[src,out,w,h,c,1,0]); assert out.read_bytes()!=src.read_bytes()
        run('optical_motion_blur',[src,out,w,h,c,1,0,0]); assert out.read_bytes()==src.read_bytes()
        run('optical_zoom_blur',[src,out,w,h,c,1,1,0]); assert out.read_bytes()==src.read_bytes()
        run('optical_chromatic_aberration',[src,out,w,h,c,0,0,0,1,1,1,0]); assert out.read_bytes()==src.read_bytes()
        run('lateral_chromatic_aberration',[src,out,w,h,c,0,0,0]); assert out.read_bytes()==src.read_bytes()
        run('longitudinal_chromatic_aberration',[src,out,w,h,c,0,0,0,1]); assert out.read_bytes()==src.read_bytes()
        run('thin_prism_distortion',[src,out,w,h,c,0,0,0,0,0]); assert out.read_bytes()==src.read_bytes()
        rx=p/'x.f32'; ry=p/'y.f32'; rx.write_bytes(struct.pack('<5f',*([0]*h))); ry.write_bytes(struct.pack('<5f',*([0]*h)))
        run('rolling_shutter_geometric_distortion',[src,out,w,h,c,rx,ry,0]); assert out.read_bytes()==src.read_bytes()
    print('PASS: optical catalog properties')
if __name__=='__main__': main()
