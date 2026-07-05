#!/usr/bin/env python3
"""Entrypoint for `lithos train` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/train_model.py` invocations keep working. Prefer `lithos train`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["train", *sys.argv[1:]]))
