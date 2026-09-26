#!/usr/bin/env python3
"""Small dependency-free properties for the explicit blend CLI contract."""
import pathlib, subprocess, sys, tempfile
exe = sys.argv[1]
with tempfile.TemporaryDirectory() as td:
    root = pathlib.Path(td); source = root/'source.raw'; overlay = root/'overlay.raw'; out = root/'out.raw'
    source.write_bytes(bytes([0, 10, 255, 7])); overlay.write_bytes(bytes([255, 20, 0, 9]))
    def run(alpha):
        subprocess.run([exe, 'blend_alpha', str(source), str(overlay), str(out), '2', '1', '2', str(alpha)], check=True)
        return out.read_bytes()
    assert run(0.0) == source.read_bytes()
    assert run(1.0) == overlay.read_bytes()
    assert run(0.5) == bytes([128, 15, 128, 8])
print('blend properties passed')
