#!/usr/bin/env python3
"""Entrypoint for `lithos tir-battery` — a shim over lithos.cli, kept so torchrun and
direct `python scripts/run_tir_battery.py` invocations keep working. Prefer
`lithos tir-battery`."""
import sys

from lithos.cli import main

if __name__ == "__main__":
    sys.exit(main(["tir-battery", *sys.argv[1:]]))
