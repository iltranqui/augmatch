#!/usr/bin/env python3
"""Statistical contract check without third-party dependencies."""
import struct, subprocess, sys, tempfile, os

def main(cli):
    t,h,w,c=48,8,8,1; n=t*h*w*c
    with tempfile.TemporaryDirectory() as d:
        i=os.path.join(d,'i');o=os.path.join(d,'o');open(i,'wb').write(struct.pack('%df'%n,*([.5]*n)))
        subprocess.check_call([cli,'temporal_gaussian',i,o,str(t),str(h),str(w),str(c),'.12','.65','123'])
        v=struct.unpack('%df'%n,open(o,'rb').read()); frame=[sum(v[k*h*w:(k+1)*h*w])/(h*w)-.5 for k in range(t)]
        a=sum(frame)/t; var=sum((x-a)*(x-a) for x in frame)/t; cov=sum((frame[k]-a)*(frame[k-1]-a) for k in range(1,t))/(t-1)
        corr=cov/var if var else 0
        if not (.00005<var<.001 and .25<corr<.95): raise AssertionError((var,corr))
if __name__=='__main__': main(sys.argv[1])
