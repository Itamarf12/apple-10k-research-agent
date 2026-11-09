"""Vector store: Chroma for search, tables.json for the full Markdown tables."""
import json
from dataclasses import dataclass
from typing import Optional

import chromadb

from config import CHROMA_DIR, COLLECTION, TABLES_PATH
from utils.embeddings import embed
from utils.indexer import Chunk

SEP = "|"        # Chroma metadata holds only simple values, so lists are stored joined by "|"


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    distance: float = 0.0

    def split(self, field: str) -> list[str]:
        return [v for v in self.meta.get(field, "").split(SEP) if v]


def _metadata(c: Chunk) -> dict:
    return {
        "type": c.type, "seq": c.seq, "item": c.item, "note": c.note,
        "heading_path": c.heading_path, "page": c.page, "citation": c.citation,
        "entities": SEP.join(c.entities), "refs": SEP.join(c.refs),
        "linked_ids": SEP.join(c.linked_ids),
    }


def build_store(chunks: list[Chunk]) -> None:
    """Write all chunks to Chroma and the Markdown tables to tables.json."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                          embedding_function=None)   # we pass embeddings ourselves
    for i in range(0, len(chunks), 64):
        batch = chunks[i:i + 64]
        collection.add(ids=[c.id for c in batch], documents=[c.text for c in batch],
                       embeddings=embed([c.text for c in batch]),
                       metadatas=[_metadata(c) for c in batch])
    tables = {c.id: c.markdown for c in chunks if c.type == "table"}
    TABLES_PATH.write_text(json.dumps(tables, indent=1), encoding="utf-8")


class Store:
    """Read access used by the document_qa tool."""

    def __init__(self) -> None:
        if not TABLES_PATH.exists() or not CHROMA_DIR.exists():
            raise FileNotFoundError("Index not built. Run: python src/build_index.py")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION, embedding_function=None)
        self.tables: dict[str, str] = json.loads(TABLES_PATH.read_text(encoding="utf-8"))

    def search(self, query: str, k: int, where: Optional[dict] = None) -> list[Hit]:
        res = self.collection.query(query_embeddings=embed([query]), n_results=k, where=where)
        return [Hit(i, d, m, dist) for i, d, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def get(self, ids: list[str]) -> list[Hit]:
        res = self.collection.get(ids=ids)
        return [Hit(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
