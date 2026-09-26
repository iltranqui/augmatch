#!/usr/bin/env python3
"""CLI smoke/property checks for deterministic ISP artifact contracts."""
import pathlib, subprocess, sys, tempfile

def main():
    exe=sys.argv[1]
    w,h,c=7,5,1
    data=bytes((x*31+y*17)%256 for y in range(h) for x in range(w))
    commands=[
      ("edge_oversharpening",[".25"]),
      ("unsharp_mask_halos",["1","1.0",".5"]),
      ("laplacian_halos",[".25"]),
      ("ringing_near_strong_edges",[".5","8","16"]),
      ("local_contrast_enhancement_artifacts",["1",".5","1"]),
      ("haloing_from_tone_mapping",["2","1.5",".5","1"]),
      ("local_sharpening_noise_amplification",["1",".5","2"]),
      ("high_frequency_attenuation",["1","1"]),
      ("detail_smearing",["1","1",".5","1"]),
      ("overshoot_and_undershoot",["1","1","1","8","8"]),
    ]
    with tempfile.TemporaryDirectory() as td:
      src=pathlib.Path(td)/"in.raw"; src.write_bytes(data)
      for op,args in commands:
        dst=pathlib.Path(td)/(op+".raw")
        subprocess.run([exe,op,str(src),str(dst),str(w),str(h),str(c),*args],check=True)
        result=dst.read_bytes()
        assert len(result)==len(data), op
        assert all(0<=v<=255 for v in result), op
        dst2=pathlib.Path(td)/(op+"-repeat.raw")
        subprocess.run([exe,op,str(src),str(dst2),str(w),str(h),str(c),*args],check=True)
        assert result==dst2.read_bytes(), op
    print("ISP artifact properties passed")

if __name__=="__main__": main()
