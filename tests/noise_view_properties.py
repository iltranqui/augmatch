#!/usr/bin/env python3
"""Small process-level property checks for the stateless noise-view contract."""
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    command = [sys.argv[1]]
    for _ in range(8):
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
