import pathlib, struct, subprocess, sys, tempfile

def write(path, values): path.write_bytes(struct.pack('<%df' % len(values), *values))
def read(path): return struct.unpack('<%df' % (path.stat().st_size//4), path.read_bytes())
def main():
    exe=sys.argv[1]; w,h,c=4,2,1; image=[.2]*(w*h*c)
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td); src=p/'in.f32'; out=p/'out.f32'; mask=p/'mask.f32'; write(src,image); write(mask,[0,1,0,1,0,1,0,1])
        for op in ('atmospheric_haze','smoke_veil','window_glare','backlight_washout'):
            subprocess.run([str(exe),op,str(src),str(out),str(w),str(h),str(c),'.5',str(mask)],check=True)
            got=read(out); assert all(0 <= x <= 1 for x in got)
        subprocess.run([str(exe),'underexposure',str(src),str(out),str(w),str(h),str(c),'.5'],check=True)
        assert all(abs(x-.1)<1e-6 for x in read(out))
        subprocess.run([str(exe),'overexposure',str(src),str(out),str(w),str(h),str(c),'3'],check=True)
        assert all(abs(x-.6)<1e-6 for x in read(out))
    print('PASS: environmental properties')
if __name__ == '__main__': main()
