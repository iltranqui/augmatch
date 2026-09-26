import pathlib, struct, subprocess, sys, tempfile

def read_f32(path): return list(struct.unpack('<%df' % (path.stat().st_size // 4), path.read_bytes()))
def write_f32(path, values): path.write_bytes(struct.pack('<%df' % len(values), *values))
def run(exe, op, args): subprocess.run([exe, op, *map(str,args)], check=True)
def close(a,b,tol=1e-6): return all(abs(x-y)<=tol for x,y in zip(a,b))

def main():
    exe=sys.argv[1]; w,h,c=4,3,2; image=[0.1+i*0.7/(w*h*c-1) for i in range(w*h*c)]
    with tempfile.TemporaryDirectory() as td:
        p=pathlib.Path(td); src=p/'in.f32'; write_f32(src,image); out=p/'out.f32'
        run(exe,'lens_vignetting',[src,out,w,h,c,0,0,0]); assert close(read_f32(out),image)
        run(exe,'optical_falloff',[src,out,w,h,c,0,-1,0]); assert min(read_f32(out))>=0
        gain=[1.0]*(w*h); gain[0]=0.25; gm=p/'map.f32'; write_f32(gm,gain)
        run(exe,'lens_shading',[src,out,w,h,c,1,gm]); got=read_f32(out); assert close(got[:c],[x*.25 for x in image[:c]])
        run(exe,'uneven_illumination',[src,out,w,h,c,1,0,gm]); assert close(read_f32(out)[:c],[x*.25 for x in image[:c]])
        run(exe,'sensor_lens_dust_shadows',[src,out,w,h,c,1,gm]); assert close(read_f32(out)[:c],[x*.75 for x in image[:c]])
    print('PASS: optical artifact properties')
if __name__=='__main__': main()
