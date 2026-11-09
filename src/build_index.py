"""Build the search index once: parse the 10-K, chunk it, and write Chroma + tables.json.

Usage: python src/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "utils" and "config" import

from config import HTML_PATH
from utils.indexer import build_chunks
from utils.parser import parse_10k
from utils.store import build_store


def main() -> None:
    if not HTML_PATH.exists():
        raise SystemExit(f"10-K not found at {HTML_PATH}. Download it from SEC EDGAR (see README).")
    blocks = parse_10k(HTML_PATH)
    chunks = build_chunks(blocks)
    n_tables = sum(c.type == "table" for c in chunks)
    print(f"Parsed {len(blocks)} blocks -> {len(chunks)} chunks ({n_tables} tables)")
    build_store(chunks)
    print("Index written to data/chroma and data/tables.json")


if __name__ == "__main__":
    main()
