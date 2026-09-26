#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile

def run(cli, op, args, data, w=16, h=1, c=1):
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d); (p/'in').write_bytes(bytes(data))
        subprocess.run([cli, 'tone', op, str(p/'in'), str(p/'out'), str(w), str(h), str(c), *map(str,args)], check=True)
        return (p/'out').read_bytes()

def main():
    cli=sys.argv[1]; data=bytes(range(16))
    assert run(cli,'gamma_variation',[1],data)==data
    assert run(cli,'s_curve_contrast_variation',[0],data)==data
    assert run(cli,'highlight_rolloff_variation',[.8,0],data)==data
    assert run(cli,'shadow_lift',[0,.25],data)==data
    assert run(cli,'shadow_crush',[0,.25],data)==data
    for op in ('posterization','low_bit_depth_banding'):
        out=run(cli,op,[3],data)
        assert all(x % 32 == 0 for x in out)
    lut=bytes(255-x for x in range(256))
    with tempfile.TemporaryDirectory() as d:
        p=pathlib.Path(d); (p/'in').write_bytes(data); (p/'lut').write_bytes(lut)
        subprocess.run([cli,'tone','tone_curve_variation',str(p/'in'),str(p/'out'),'16','1','1',str(p/'lut')],check=True)
        assert (p/'out').read_bytes()==bytes(255-x for x in data)
    print('tone variation properties passed')
if __name__ == '__main__': main()
