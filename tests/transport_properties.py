#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile

def main(exe):
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td); src=root/'in'; dst=root/'out'; src.write_bytes(bytes(range(16)))
        subprocess.check_call([exe,'truncated_frame',str(src),str(dst),'16','7','255'])
        got=dst.read_bytes(); assert got[:7]==bytes(range(7)); assert got[7:]==b'\xff'*9
        subprocess.check_call([exe,'bit_flips',str(src),str(dst),'16','0','1','4'])
        assert dst.read_bytes()==bytes(range(16))
    print('transport properties passed')
if __name__ == '__main__': main(sys.argv[1])
