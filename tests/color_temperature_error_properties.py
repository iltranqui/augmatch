#!/usr/bin/env python3
import pathlib, subprocess, sys, tempfile


def main():
    exe = sys.argv[1]
    width, height, channels = 11, 7, 4
    data = bytes((x * 29 + y * 43 + channel * 67) % 256
                 for y in range(height) for x in range(width) for channel in range(channels))
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        src = root / "in.raw"
        src.write_bytes(data)
        # Identity cases: zero strength is a no-op at every valid temperature;
        # the 6500 K reference is a no-op at every valid strength.
        for temperature, strength in ((1000, 0), (3000, 0), (6500, 0.3), (40000, 1)):
            output = root / f"{temperature}-{strength}.raw"
            command = [exe, "color_temperature_error", str(src), str(output), str(width), str(height), str(channels), str(temperature), str(strength)]
            subprocess.run(command, check=True)
            result = output.read_bytes()
            assert len(result) == len(data)
            if strength == 0 or temperature == 6500:
                assert result == data
            assert all(result[i] == data[i] for i in range(3, len(data), channels))
            repeated = root / "repeat.raw"
            subprocess.run(command[:3] + [str(repeated)] + command[4:], check=True)
            assert repeated.read_bytes() == result
    print("color temperature error properties passed")


if __name__ == "__main__":
    main()
