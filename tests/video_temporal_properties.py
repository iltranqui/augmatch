#!/usr/bin/env python3
"""CLI smoke/property checks for the eight deterministic video artifacts."""
import os, struct, subprocess, sys, tempfile

def run(cli, op, inp, out, *args):
    subprocess.check_call([cli, op, inp, out, *map(str, args)])
    return struct.unpack("%df" % (len(open(out, "rb").read()) // 4), open(out, "rb").read())

def main(cli):
    t,h,w,c=4,2,3,1; n=t*h*w*c; data=[i/float(n) for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        i=os.path.join(d,"in.f32"); o=os.path.join(d,"out.f32")
        open(i,"wb").write(struct.pack("%df"%n,*data))
        mask=os.path.join(d,"mask.raw"); open(mask,"wb").write(bytes([0,1,0,0,0,0]))
        run(cli,"dead_pixel_persistence",i,o,t,h,w,c,.1,.2,7,mask)
        vals=run(cli,"hot_pixel_persistence",i,o,t,h,w,c,.9,.2,7,mask)
        assert all(0 <= x <= 1 for x in vals)
        drop=os.path.join(d,"drop.raw"); open(drop,"wb").write(bytes([0,1,0,1]))
        vals=run(cli,"frame_drops",i,o,t,h,w,c,0,.0,7,drop); assert all(abs(a-b)<1e-6 for a,b in zip(vals[w*h*c:2*w*h*c],data[:w*h*c]))
        idx=os.path.join(d,"idx.i32"); open(idx,"wb").write(struct.pack("4i",0,0,2,1))
        vals=run(cli,"duplicate_frames",i,o,t,h,w,c,0,7,idx); assert all(abs(a-b)<1e-6 for a,b in zip(vals[w*h*c:2*w*h*c],data[:w*h*c]))
        vals=run(cli,"frame_blending",i,o,t,h,w,c,.5); assert len(vals)==n
        vals=run(cli,"temporal_ghosting",i,o,t,h,w,c,0,1,0,.5); assert len(vals)==n
        vals=run(cli,"motion_compensation_errors",i,o,t,h,w,c,-1,0,0,.5); assert len(vals)==n
        vals=run(cli,"video_sensor_rolling_shutter",i,o,t,h,w,c,2,0,1); assert len(vals)==n
        assert all(0 <= x <= 1 for x in vals)

if __name__ == "__main__": main(sys.argv[1])
