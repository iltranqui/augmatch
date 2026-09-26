#!/usr/bin/env python3
"""Property/reference checks for YCbCr subsampling and local tone noise."""
import pathlib, subprocess, sys, tempfile

def run(cli, op, data, args):
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d); (p/'in').write_bytes(bytes(data))
        subprocess.run([cli,op,str(p/'in'),str(p/'out'),*map(str,args)], check=True)
        return (p/'out').read_bytes()

def main():
    w,h,c=4,2,4
    data=bytes((x*17+y*31+k*43)%256 for y in range(h) for x in range(w) for k in range(c))
    assert run(sys.argv[1], 'chroma_subsampling_artifacts', data, [w,h,c,'444']) == data
    for mode in ('422','420'):
        out=run(sys.argv[1], 'chroma_subsampling_artifacts', data, [w,h,c,mode])
        assert len(out)==len(data) and out[3::4]==data[3::4]
    # Reference a single 4:2:2 pair: Y is retained while the pair shares Cb/Cr.
    pair=bytes((220,40,20,9,20,180,230,10))
    actual=list(run(sys.argv[1], 'chroma_subsampling_artifacts', pair, [2,1,4,'422']))
    ycc=[]
    for r,g,b in ((220,40,20),(20,180,230)):
        y=.299*r/255+.587*g/255+.114*b/255
        ycc.append((y,b/255-y,r/255-y))
    cb=sum(v[1] for v in ycc)/2; cr=sum(v[2] for v in ycc)/2
    expected=[]
    for y,_,_ in ycc:
        r=y+cr; b=y+cb; g=(y-.299*r-.114*b)/.587
        expected.extend([max(0,min(255,int(255*r+.5))), max(0,min(255,int(255*g+.5))), max(0,min(255,int(255*b+.5)))])
    assert actual[:3]+actual[4:7] == expected
    args=[w,h,c,1,.35,.015,77]
    first=run(sys.argv[1], 'local_tone_mapping_noise', data, args)
    second=run(sys.argv[1], 'local_tone_mapping_noise', data, args)
    assert first==second and len(first)==len(data) and first[3::4]==data[3::4]
    identity=run(sys.argv[1], 'local_tone_mapping_noise', data, [w,h,c,0,0,0,0])
    # Round-tripping BT.601 with zero parameters is bounded to one uint8 level.
    assert max(abs(a-b) for a,b in zip(identity,data)) <= 1
    print('color pipeline artifact properties passed')

if __name__ == '__main__':
    main()
