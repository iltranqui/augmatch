import pathlib, struct, subprocess, sys, tempfile

def pack(fmt, rows): return b''.join(struct.pack(fmt, *row) for row in rows)
def write(path, values): path.write_bytes(struct.pack('<%df' % len(values), *values))
def read(path): return struct.unpack('<%df' % (path.stat().st_size//4), path.read_bytes())
def main():
    exe=sys.argv[1]; w,h,c=5,3,1; image=[0.1]*(w*h)
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td); src=p/'in.f32'; out=p/'out.f32'; write(src,image)
        (p/'sources.bin').write_bytes(pack('<ffffi',[(2,1,.1,0,-1)]))
        (p/'halos.bin').write_bytes(pack('<ffffi',[(2,1,2,.5,-1)]))
        subprocess.run([str(exe),'flare',str(src),str(out),str(w),str(h),str(c),'1',str(p/'sources.bin'),str(p/'halos.bin')],check=True)
        got=read(out); assert abs(got[7]-.55)<1e-6 and abs(got[0]-image[0])<1e-6
        (p/'ghosts.bin').write_bytes(pack('<ffff',[(1,0,1,1)]))
        write(src,[0.8]+[0.1]*(w*h-1)); subprocess.run([str(exe),'ghosting',str(src),str(out),str(w),str(h),str(c),'1',str(p/'ghosts.bin')],check=True)
        assert abs(read(out)[1]-.8)<1e-6
        subprocess.run([str(exe),'veiling_glare',str(src),str(out),str(w),str(h),str(c),'.5'],check=True); assert abs(read(out)[0]-1.0)<1e-6
        write(src,[1.0 if i==7 else 0.0 for i in range(w*h)])
        subprocess.run([str(exe),'bloom',str(src),str(out),str(w),str(h),str(c),'.5','1','1'],check=True); assert read(out)[6]>0
    print('PASS: optical scatter properties')
if __name__ == '__main__': main()
