#!/usr/bin/env python3
"""Run measured CPU noise benchmarks and inventory camera-domain fixtures.

The script intentionally does not synthesize a profile dataset.  A missing
fixture is reported as a runtime-gated partial row, not as a passing result.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: noise_catalog_benchmarks.py NOISE_CATALOG_VALIDATION", file=sys.stderr)
        return 2
    executable = Path(sys.argv[1])
    if not executable.exists():
        print(f"BLOCKED: validation executable is missing: {executable}", file=sys.stderr)
        return 2
    print(f"PYTHON_VERSION={sys.version.split()[0]}")
    result = subprocess.run([str(executable), "--benchmark"], check=False, text=True, capture_output=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    expected = {"512x512", "1920x1080", "3840x2160"}
    measured = {line.split()[1] for line in result.stdout.splitlines() if line.startswith("BENCHMARK ")}
    if measured != expected:
        print(f"BLOCKED: benchmark sizes {sorted(measured)} != {sorted(expected)}", file=sys.stderr)
        return 1
    for line in result.stdout.splitlines():
        if line.startswith("BENCHMARK ") and not re.search(r"transform_ms=[0-9.eE+-]+ memcpy_ms=[0-9.eE+-]+ bytes=[0-9]+", line):
            print(f"BLOCKED: malformed measured benchmark line: {line}", file=sys.stderr)
            return 1

    root = Path(__file__).resolve().parents[2]
    data = root / "data"
    profiles = []
    if data.is_dir():
        for path in sorted(data.rglob("*")):
            if path.is_file() and ("profile" in path.name.lower() or path.suffix.lower() in {".profile", ".cam", ".cube"}):
                profiles.append(path)
    if profiles:
        print("PROFILE_DATASET=FOUND")
        for path in profiles:
            print(f"PROFILE_FIXTURE={path.relative_to(root)}")
        print("PROFILE_REGRESSION=NOT_RUN blocker=fixture format is not a native camera profile")
    else:
        print("PROFILE_DATASET=SKIP blocker=no camera-domain profile fixtures in data/")
    print("PASS: measured noise benchmarks and profile-fixture inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
