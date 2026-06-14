#!/usr/bin/env python
"""Move artifacts between local disk and the durable store (PRD §26.5).

# push tokenized shards to the object store
python scripts/sync.py push data/fineweb-edu/corpus-v0.1 corpus/fineweb-edu-v0.1

# pull them onto a GPU node before training
python scripts/sync.py pull corpus/fineweb-edu-v0.1 data/fineweb-edu/corpus-v0.1
"""

from __future__ import annotations

import argparse

from lithos.utils.storage import load_storage


def main() -> None:
    ap = argparse.ArgumentParser(description="Push/pull artifacts to/from the durable store.")
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("src", help="Source: local path (push) or store path (pull).")
    ap.add_argument("dst", help="Destination: store path (push) or local path (pull).")
    ap.add_argument("--storage-config", default="configs/storage.yaml")
    args = ap.parse_args()

    storage = load_storage(args.storage_config)
    if args.action == "push":
        dest = storage.put(args.src, args.dst)  # put(local, rel)
        print(f"pushed {args.src} -> {dest}")
    else:
        storage.get(args.src, args.dst)  # get(rel, local)
        print(f"pulled {storage.uri(args.src)} -> {args.dst}")


if __name__ == "__main__":
    main()
