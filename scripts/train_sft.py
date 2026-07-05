#!/usr/bin/env python3
"""Entrypoint for `lithos sft` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/train_sft.py` invocations keep working. Prefer `lithos sft`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["sft", *sys.argv[1:]]))
