#!/usr/bin/env python3
"""Entrypoint for `lithos eval` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/run_evals.py` invocations keep working. Prefer `lithos eval`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
