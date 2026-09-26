#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile

exe = sys.argv[1]
w, h, c = 5, 3, 3
source = bytes((x * 17 + 3) % 256 for x in range(w * h * c))
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td)
    inp, out1, out2 = (root / name for name in ("in.raw", "one.raw", "two.raw"))
    inp.write_bytes(source)
    def run(*args):
        subprocess.check_call([exe, *map(str, args)])
    run("speckle_noise", inp, out1, w, h, c, 0, 0.1, 19)
    run("speckle_noise", inp, out2, w, h, c, 0, 0.1, 19)
    assert out1.read_bytes() == out2.read_bytes()
    run("fog", inp, out1, w, h, c, 0, 1, 19)
    assert out1.read_bytes() == source
    run("contrast", inp, out1, w, h, c, 1)
    assert out1.read_bytes() == source
    run("brightness", inp, out1, w, h, c, 0)
    assert set(out1.read_bytes()) == {0}
    run("pixelate", inp, out1, w, h, c, 1)
    assert out1.read_bytes() == source
print("imgcorruptlike properties passed")
