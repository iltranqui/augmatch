#!/usr/bin/env python3
import pathlib,subprocess,sys,tempfile,statistics

def main():
 exe=sys.argv[1];w=h=16;c=1;src_data=bytes([128])*(w*h*c)
 with tempfile.TemporaryDirectory() as td:
  p=pathlib.Path(td);src=p/'in.raw';src.write_bytes(src_data);means=[]
  for seed in range(8):
   out=p/f'{seed}.raw';subprocess.run([exe,'gradient_plus_laplacian_noise',str(src),str(out),str(w),str(h),str(c),'4','0','0',str(seed)],check=True);means.append(statistics.mean(out.read_bytes()))
  assert max(means)-min(means)>0, means
 print('remaining noise statistics passed')
if __name__=='__main__':main()
