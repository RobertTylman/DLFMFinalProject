#!/usr/bin/env python3
"""Index the GTZAN originals into a dedicated fingerprint database.

Keeps the main `fingerprints.db` untouched. The resulting `fingerprints_gtzan.db`
contains only the 1,000 originals (minus jazz.00054.wav, which is corrupt and
silently skipped by the indexer).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.index_directory import index_folder


DEFAULT_ORIGINALS = "/Volumes/Robbie SSD/GTZAN Dataset/Data/genres_original"
DEFAULT_DB = str(Path(__file__).resolve().parent / "fingerprints_gtzan.db")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--originals-root", default=DEFAULT_ORIGINALS,
                   help=f"GTZAN originals root (default: {DEFAULT_ORIGINALS})")
    p.add_argument("--db", default=DEFAULT_DB,
                   help=f"Output database path (default: {DEFAULT_DB})")
    args = p.parse_args()

    if not os.path.isdir(args.originals_root):
        sys.exit(f"Not a directory: {args.originals_root}")

    print(f"Indexing originals from: {args.originals_root}")
    print(f"Writing fingerprint DB to: {args.db}")
    index_folder(args.originals_root, db_path=args.db)


if __name__ == "__main__":
    main()
