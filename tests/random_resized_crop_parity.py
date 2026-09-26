#!/usr/bin/env python3
"""Exact/property checks for the explicit RandomResizedCrop/SizedCrop contract."""
import pathlib, subprocess, sys, tempfile
MASK=(1<<64)-1

def splitmix64(value):
    value=(value+0x9E3779B97F4A7C15)&MASK
    value=((value^(value>>30))*0xBF58476D1CE4E5B9)&MASK
    value=((value^(value>>27))*0x94D049BB133111EB)&MASK
    return (value^(value>>31))&MASK

def resize(src, iw, ih, ch, ow, oh, mode):
    out=bytearray(ow*oh*ch); sx=iw/ow; sy=ih/oh
    def at(x,y,c): return src[(y*iw+x)*ch+c]
    for y in range(oh):
      for x in range(ow):
        for c in range(ch):
          if mode==0:
            ix=min(iw-1,int(x*sx)); iy=min(ih-1,int(y*sy)); v=at(ix,iy,c)
          else:
            fx=max(0,min(iw-1,(x+.5)*sx-.5)); fy=max(0,min(ih-1,(y+.5)*sy-.5))
            x0=int(fx); y0=int(fy); x1=min(iw-1,x0+1); y1=min(ih-1,y0+1)
            ax=fx-x0; ay=fy-y0
            v=round((1-ay)*((1-ax)*at(x0,y0,c)+ax*at(x1,y0,c))+ay*((1-ax)*at(x0,y1,c)+ax*at(x1,y1,c)))
          out[(y*ow+x)*ch+c]=max(0,min(255,v))
    return bytes(out)

def generated_rect(w, h, seed):
    cw=1+splitmix64(seed)%w; ch=1+splitmix64(seed+1)%h
    x=splitmix64(seed+2)%(w-cw+1); y=splitmix64(seed+3)%(h-ch+1)
    return x,y,cw,ch

def run(exe, op, image, args, td):
    src=pathlib.Path(td)/'in.raw'; dst=pathlib.Path(td)/'out.raw'; src.write_bytes(image)
    subprocess.run([exe,op,str(src),str(dst)]+[str(v) for v in args],check=True)
    return dst.read_bytes()

def main():
    exe=sys.argv[1]; op=sys.argv[2] if len(sys.argv)>2 else 'random_resized_crop'
    w,h,ch=9,7,2; ow,oh=5,4
    image=bytes((i*31+7)&255 for i in range(w*h*ch))
    with tempfile.TemporaryDirectory() as td:
      # Explicit rectangle exercises nearest and the exact resize contract.
      args=[w,h,ch,ow,oh,2,1,5,4,0]
      expected=bytearray()
      for y in range(1,5): expected.extend(image[(y*w+2)*ch:(y*w+7)*ch])
      assert run(exe,op,image,args,td)==resize(bytes(expected),5,4,ch,ow,oh,0)
      args[-1]=1
      assert run(exe,op,image,args,td)==resize(bytes(expected),5,4,ch,ow,oh,1)
      # Zero dimensions and -1 offsets are deterministic seed generation.
      generated=run(exe,op,image,[w,h,ch,ow,oh,-1,-1,0,0,1,1234],td)
      assert len(generated)==ow*oh*ch
      assert generated==run(exe,op,image,[w,h,ch,ow,oh,-1,-1,0,0,1,1234],td)
      x,y,cw,hh=generated_rect(w,h,1234); source=bytearray()
      for row in range(y,y+hh): source.extend(image[(row*w+x)*ch:(row*w+x+cw)*ch])
      assert generated==resize(bytes(source),cw,hh,ch,ow,oh,1)
      # Invalid interpolation and out-of-bounds explicit rectangles fail.
      bad=subprocess.run([exe,op,str(pathlib.Path(td)/'in.raw'),str(pathlib.Path(td)/'bad.raw'),str(w),str(h),str(ch),str(ow),str(oh),"8","5","2","2","0"],stderr=subprocess.DEVNULL)
      assert bad.returncode!=0
    print('PASS: '+op+' exact resize parity, deterministic generation, and properties')
if __name__=='__main__': main()
