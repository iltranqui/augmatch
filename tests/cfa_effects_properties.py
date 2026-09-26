#!/usr/bin/env python3
import struct, subprocess, sys, tempfile

def main():
    exe = sys.argv[1]; w = h = 32; n = w*h
    values = [0.5] * n
    with tempfile.TemporaryDirectory() as d:
        src, dst = d + "/in.bin", d + "/out.bin"
        open(src, "wb").write(struct.pack("<%sf" % n, *values))
        subprocess.check_call([exe, "bayer_plane_gain", src, dst, str(w), str(h), "1", "0.5", "0.25", "0.75"])
        got = struct.unpack("<%sf" % n, open(dst, "rb").read())
        expected = [0.5, 0.25, 0.125, 0.375]
        for y in range(h):
            for x in range(w):
                if abs(got[y*w+x] - expected[((y & 1) << 1) | (x & 1)]) > 1e-6:
                    raise AssertionError("plane gain map property failed")
        subprocess.check_call([exe, "bayer_plane_noise", src, dst, str(w), str(h), "0.02", "0.02", "0.02", "0.02"])
        noise = struct.unpack("<%sf" % n, open(dst, "rb").read())
        if not all(0.0 <= x <= 1.0 for x in noise): raise AssertionError("noise clipping property failed")
        mean = sum(noise) / n
        variance = sum((x - mean)*(x - mean) for x in noise) / n
        if abs(mean - 0.5) > 0.03 or variance <= 0.00005: raise AssertionError("noise statistical contract failed")
    print("PASS: CFA effects CLI properties and noise statistics")

if __name__ == "__main__": main()
