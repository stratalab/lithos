#!/usr/bin/env python3
"""Entrypoint for `lithos grpo` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/train_grpo.py` invocations keep working. Prefer `lithos grpo`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["grpo", *sys.argv[1:]]))
