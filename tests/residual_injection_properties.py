#!/usr/bin/env python3
"""CLI exact/property checks for residual-map ISP injections."""
import pathlib, struct, subprocess, sys, tempfile

def main():
    exe=sys.argv[1]; w,h,c=4,3,2
    data=bytes((17*i+3)%256 for i in range(w*h*c))
    values=[float((i%7)-3) for i in range(w*h*c)]
    packed=struct.pack('<'+'f'*len(values), *values)
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); src=td/'in.raw'; src.write_bytes(data); mp=td/'map.f32'; mp.write_bytes(packed)
        commands=[('laplacian_residual_injection',['0.5']),('high_pass_residual_injection',['0.5'])]
        for op,args in commands:
            for suffix,extra in [('derived',[]),('mapped',[str(mp)])]:
                dst=td/(op+suffix+'.raw')
                subprocess.run([exe,op,str(src),str(dst),str(w),str(h),str(c),*args,*extra],check=True)
                result=dst.read_bytes(); assert len(result)==len(data); assert len(set(result))>1
            zero=td/(op+'zero.raw')
            subprocess.run([exe,op,str(src),str(zero),str(w),str(h),str(c),'0'],check=True)
            assert zero.read_bytes()==data
        dst=td/'sobel.raw'; subprocess.run([exe,'sobel_residual_injection',str(src),str(dst),str(w),str(h),str(c),'.25','0',str(mp)],check=True)
        assert len(dst.read_bytes())==len(data)
    print('Residual injection properties passed')

if __name__ == '__main__': main()
