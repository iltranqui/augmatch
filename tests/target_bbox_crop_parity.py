#!/usr/bin/env python3
"""Reference checks for deterministic target-aware crop contracts."""
import pathlib, subprocess, tempfile
MASK=(1<<64)-1
def splitmix(v):
    v=(v+0x9E3779B97F4A7C15)&MASK; v=((v^(v>>30))*0xBF58476D1CE4E5B9)&MASK; v=((v^(v>>27))*0x94D049BB133111EB)&MASK; return (v^(v>>31))&MASK
def between(a,b,s): return a+splitmix(s)%(b-a+1)
def crop(img,W,C,x,y,w,h):
    return b''.join(img[(yy*W+x)*C:(yy*W+x+w)*C] for yy in range(y,y+h))
def boxes_file(td,boxes):
    p=pathlib.Path(td)/"boxes.txt"; p.write_text(''.join('%s %s %s %s\n'%b for b in boxes)); return p
def run(exe,args): return subprocess.run([exe]+args,check=True,text=True,capture_output=True)
def main():
    exe=sys.argv[1]; W,H,C=17,13,2; img=bytes((i*17+3)&255 for i in range(W*H*C)); boxes=[(3.,2.,9.,7.),(11.,5.,15.,11.)]
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); src=td/'in.raw'; src.write_bytes(img); bf=boxes_file(td,boxes)
        # Safe crop uses the union envelope and min dimensions, with seed offsets 0..3.
        out=td/'safe.raw'; r=run(exe,['bbox_safe_random_crop',str(src),str(out),str(W),str(H),str(C),str(bf),'1','1','0','0','0','77'])
        w=between(12,17,77); h=between(9,13,78); x=between(max(0,15-w),min(3,W-w),79); y=between(max(0,11-h),min(2,H-h),80)
        assert r.stdout.strip()==f'{x} {y} {w} {h} 0'; assert out.read_bytes()==crop(img,W,C,x,y,w,h)
        # Fixed output safe crop has the same source selection and reports scale metadata.
        out=td/'sized.raw'; r=run(exe,['random_sized_bbox_safe_crop',str(src),str(out),str(W),str(H),str(C),str(bf),'5','4','0','0','77'])
        assert r.stdout.split()[:4]==[str(x),str(y),str(w),str(h)]; assert len(out.read_bytes())==5*4*C
        # AtLeastOne selects a target with seed+0 and chooses an origin containing it.
        out=td/'one.raw'; r=run(exe,['at_least_one_bbox_random_crop',str(src),str(out),str(W),str(H),str(C),str(bf),'6','5','0','-1','-1','77'])
        selected=splitmix(77)%len(boxes); b=boxes[selected]; x1=int(b[0]); y1=int(b[1]); x2=int(__import__('math').ceil(b[2])); y2=int(__import__('math').ceil(b[3]))
        ax=between(max(0,x1-6+1),min(x2-1,W-6),78); ay=between(max(0,y1-5+1),min(y2-1,H-5),79)
        assert r.stdout.split()[:4]==[str(ax),str(ay),'6','5']; assert out.read_bytes()==crop(img,W,C,ax,ay,6,5)
        # Empty arrays exercise explicit and seeded fallback paths.
        empty=boxes_file(td,[]); out=td/'fallback.raw'; r=run(exe,['at_least_one_bbox_random_crop',str(src),str(out),str(W),str(H),str(C),str(empty),'4','3','0','2','4','88'])
        assert r.stdout.strip()=='2 4 4 3 18446744073709551615 1'; assert out.read_bytes()==crop(img,W,C,2,4,4,3)
    print('PASS: target-aware crop deterministic selection, safe envelopes, resize, and fallback')
if __name__=='__main__':
    import sys
    main()
