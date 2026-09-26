import struct, subprocess, sys, tempfile
from pathlib import Path

def run(exe, op, args):
    subprocess.run([exe, op, *map(str,args)], check=True)

def main():
    exe=sys.argv[1]
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); w,h,c=7,5,1; src=p/'in.raw'; src.write_bytes(bytes((x+31*y)%256 for y in range(h) for x in range(w)))
        depth=p/'depth.f32'; depth.write_bytes(struct.pack('<%df'%(w*h), *([.5]*(w*h))))
        out=p/'out.raw'; run(exe,'depth_defocus',[src,out,w,h,c,depth,.5,10,8]); assert out.read_bytes()==src.read_bytes()
        x=p/'x.f32'; y=p/'y.f32'; x.write_bytes(struct.pack('<3f',0,0,0)); y.write_bytes(struct.pack('<3f',0,0,0))
        run(exe,'camera_shake',[src,out,w,h,c,x,y,3]); assert out.read_bytes()==src.read_bytes()
        run(exe,'linear_directional',[src,out,w,h,c,3,0,1]); assert out.read_bytes()==src.read_bytes()
        run(exe,'rotational_motion',[src,out,w,h,c,3,2,90,1]); assert out.read_bytes()==src.read_bytes()
        rows=p/'rows.f32'; rows.write_bytes(struct.pack('<5f',0,0,0,0,0)); run(exe,'rolling_shutter',[src,out,w,h,c,rows,rows,1]); assert out.read_bytes()==src.read_bytes()
    print('PASS: optical motion properties')
if __name__=='__main__': main()
