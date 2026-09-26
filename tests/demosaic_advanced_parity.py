#!/usr/bin/env python3
"""Deterministic raw-plane and CFA ownership properties for advanced demosaicers."""
import pathlib, subprocess, sys, tempfile
exe = sys.argv[1]
w, h = 9, 8
raw = bytes(((i * 37 + (i // w) * 11) & 255) for i in range(w * h))
patterns = [((0, 1), (1, 2)), ((2, 1), (1, 0)), ((1, 0), (2, 1)), ((1, 2), (0, 1))]
def ch(tile, y, x): return tile[max(0, min(h - 1, y)) & 1][max(0, min(w - 1, x)) & 1]
def at(values, x, y): return values[max(0, min(w - 1, x)) + w * max(0, min(h - 1, y))]
def round_u8(v): return max(0, min(255, int(v + .5)))
def mhc_ref(values, tile):
    green = [[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]]
    same = [[0,0,.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,.5,0,0]]
    diagonal = [[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]]
    result = []
    for y in range(h):
      for x in range(w):
       for target in range(3):
        center = ch(tile,y,x)
        if center == target: value = at(values,x,y)
        else:
         kind, horizontal = 0, True
         if target != 1:
          if center == 1:
           horizontal = ch(tile,y,x-1) == target or ch(tile,y,x+1) == target; kind = 1
          else: kind = 2
         kernel = green if kind == 0 else diagonal if kind == 2 else (same if horizontal else [list(row) for row in zip(*same)])
         value = sum(kernel[ky+2][kx+2] * at(values,x+kx,y+ky) for ky in range(-2,3) for kx in range(-2,3)) / 8
        result.append(round_u8(value))
    return bytes(result)
with tempfile.TemporaryDirectory() as td:
    src = pathlib.Path(td) / "in.raw"
    src.write_bytes(raw)
    for op in ("demosaic_malvar", "demosaic_edge_aware"):
        for pattern, tile in enumerate(patterns):
            one = pathlib.Path(td) / f"{op}-{pattern}-a.raw"
            two = pathlib.Path(td) / f"{op}-{pattern}-b.raw"
            subprocess.run([exe, op, str(src), str(one), str(w), str(h), str(pattern)], check=True)
            subprocess.run([exe, op, str(src), str(two), str(w), str(h), str(pattern)], check=True)
            got = one.read_bytes()
            assert got == two.read_bytes(), (op, pattern, "nondeterministic")
            assert len(got) == w * h * 3 and all(v <= 255 for v in got)
            if op == "demosaic_malvar":
                assert got == mhc_ref(raw, tile), (op, pattern, "coefficient reference mismatch")
            for y in range(h):
                for x in range(w):
                    channel = tile[y & 1][x & 1]
                    assert got[(y * w + x) * 3 + channel] == raw[y * w + x]
    constant = pathlib.Path(td) / "constant.raw"
    constant.write_bytes(bytes([73]) * (w * h))
    for op in ("demosaic_malvar", "demosaic_edge_aware"):
        out = pathlib.Path(td) / f"{op}-constant.raw"
        subprocess.run([exe, op, str(constant), str(out), str(w), str(h), "0"], check=True)
        assert out.read_bytes() == bytes([73]) * (w * h * 3)
print("PASS: advanced demosaicing deterministic/reference properties")
