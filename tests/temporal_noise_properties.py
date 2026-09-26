#!/usr/bin/env python3
"""CLI smoke/property checks for contiguous T-H-W-C temporal contracts."""
import os, struct, subprocess, sys, tempfile

def main(cli):
    t,h,w,c=5,2,3,2; n=t*h*w*c
    data=struct.pack("%df"%n,*([.35]*n))
    with tempfile.TemporaryDirectory() as d:
        i=os.path.join(d,"in.f32");o=os.path.join(d,"out.f32")
        open(i,"wb").write(data)
        for op,args in (("temporal_gaussian",[".1",".6","9"]),("temporal_shot",["100",".4","10"]),("temporal_read_noise",[".05",".3","11"]),("flicker",[".1",".5","12"]),("exposure_flicker",[".1",".5","12"]),("gain_flicker",[".1",".5","12"]),("fixed_pattern_noise_drift",[".02",".01",".7","13"])):
            subprocess.check_call([cli,op,i,o,str(t),str(h),str(w),str(c),*args])
            raw=open(o,"rb").read(); assert len(raw)==4*n,(op,len(raw))
            vals=struct.unpack("%df"%n,raw); assert all(0<=x<=1 for x in vals),op
        subprocess.check_call([cli,"white_balance_flicker",i,o,str(t),str(h),str(w),str(c),".1",".05",".08",".4","14"])
        assert len(open(o,"rb").read())==4*n
if __name__=="__main__": main(sys.argv[1])
