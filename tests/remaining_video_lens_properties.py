#!/usr/bin/env python3
import os, struct, subprocess, sys, tempfile

def run(cli, op, args, n, d, root):
    i=os.path.join(root,op+'.in'); o=os.path.join(root,op+'.out')
    open(i,'wb').write(struct.pack('%df'%n,*([d]*n)))
    subprocess.check_call([cli,op,i,o,*map(str,args)])
    raw=open(o,'rb').read(); assert len(raw)==4*n
    vals=struct.unpack('%df'%n,raw); assert all(0<=x<=1 for x in vals)

def main(cli):
    with tempfile.TemporaryDirectory() as d:
        run(cli,'random_telegraph_signal_noise',[4,2,3,0.1,0.2,0.4,0,7],24,.3,d)
        run(cli,'inter_frame_compression_noise',[3,2,4,1,0.1,1,9],24,.3,d)
        run(cli,'gop_keyframe_artifacts',[3,2,4,1,2,.2,.1,10],24,.3,d)
        run(cli,'block_motion_estimation_artifacts',[3,2,4,1,2,2,.5,11],24,.3,d)
        run(cli,'water_droplets_on_lens',[4,2,3,2,.7,1,2,1,12],24,.3,d)
if __name__=='__main__': main(sys.argv[1])
