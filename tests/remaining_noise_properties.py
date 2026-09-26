#!/usr/bin/env python3
import pathlib,struct,subprocess,sys,tempfile

def main():
 exe=sys.argv[1]; w,h,c=9,7,3; data=bytes(((x*23+y*17+k*31)%256) for y in range(h) for x in range(w) for k in range(c))
 ops=[('gradient_plus_laplacian_noise',['2','1','1','7']),('deblocking_halos',['8','1','.5']),('demosaicing_edge_artifacts',['.5','1']),('edge_dependent_quantization',['8','1']),('edge_dependent_compression_error',['8','1','1','7']),('gradient_reversal',['.5','1']),('clipped_edge_ringing',['.5','1','16']),('per_channel_gain_noise',['1','1','1','.02','5']),('color_temperature_error',['3000','.5'])]
 with tempfile.TemporaryDirectory() as td:
  p=pathlib.Path(td); src=p/'in.raw';src.write_bytes(data)
  for op,args in ops:
   out=p/(op+'.raw');cmd=[exe,op,str(src),str(out),str(w),str(h),str(c),*args];subprocess.run(cmd,check=True);got=out.read_bytes();assert len(got)==len(data);repeat=p/(op+'-repeat.raw');subprocess.run(cmd[:3]+[str(repeat),*cmd[4:]],check=True);assert got==repeat.read_bytes()
  floats=struct.pack('<'+('f'*(2*3*4*2)),*([.5]*(2*3*4*2)));fi=p/'in.f32';fi.write_bytes(floats);fo=p/'out.f32';subprocess.run([exe,'row_noise_phase_changes',str(fi),str(fo),'2','3','4','2','.1','.5','0','4'],check=True);assert len(fo.read_bytes())==len(floats)
 print('remaining noise properties passed')
if __name__=='__main__':main()
