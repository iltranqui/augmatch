import pathlib, struct, subprocess, sys, tempfile

def write(path, values): path.write_bytes(struct.pack('<%df' % len(values), *values))
def read(path): return struct.unpack('<%df' % (path.stat().st_size//4), path.read_bytes())
def run(exe, args): subprocess.run([str(exe), *map(str,args)], check=True)
def main():
    exe=sys.argv[1]; w=h=3; c=1; t=3; image=[.2]*(w*h); batch=[.5]*(t*w*h)
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td); src=p/'image.f32'; seq=p/'seq.f32'; out=p/'out.f32'; mask=p/'mask.f32'
        write(src,image); write(seq,batch); write(mask,[1.0]*(w*h))
        run(exe,['dirty_lens_blur',src,out,w,h,c,1,1,mask]); assert all(0<=v<=1 for v in read(out))
        run(exe,['electromagnetic_interference',src,out,w,h,c,.1,1,1,0,mask]); assert all(0<=v<=1 for v in read(out))
        run(exe,['sensor_temperature_drift',seq,out,t,h,w,c,25,35,25,100,1000,1,0.05]); assert len(read(out))==len(batch)
        run(exe,['power_supply_banding',seq,out,t,h,w,c,.2,.5,1,1,0]); assert read(out)!=tuple(batch)
        run(exe,['fluorescent_light_flicker',seq,out,t,h,w,c,.2,.5,1,0]); assert all(0<=v<=1 for v in read(out))
        run(exe,['led_rolling_band_artifacts',seq,out,t,h,w,c,.2,.5,1,1,0]); assert all(0<=v<=1 for v in read(out))
    print('PASS: acquisition artifact properties')
if __name__ == '__main__': main()
