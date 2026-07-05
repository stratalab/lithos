#!/usr/bin/env python3
"""Entrypoint for `lithos dpo` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/train_dpo.py` invocations keep working. Prefer `lithos dpo`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["dpo", *sys.argv[1:]]))
