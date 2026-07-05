#!/usr/bin/env python3
"""Entrypoint for `lithos tokenizer` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/train_tokenizer.py` invocations keep working. Prefer `lithos tokenizer`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["tokenizer", *sys.argv[1:]]))
