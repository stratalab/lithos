#!/usr/bin/env python3
"""Entrypoint for `lithos tokenize` — a shim over lithos.cli, kept so torchrun and direct
`python scripts/tokenize_corpus.py` invocations keep working. Prefer `lithos tokenize`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["tokenize", *sys.argv[1:]]))
