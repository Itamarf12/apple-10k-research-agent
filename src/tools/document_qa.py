"""Document QA tool: search Apple's 10-K and return cited passages and whole tables."""
import logging
import re
from typing import Optional

from config import MAX_CHUNKS, MAX_DISTANCE, TOP_K
from utils.store import Hit, Store

logger = logging.getLogger(__name__)
_store: Optional[Store] = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _follow_reference(store: Store, ref: str, query: str) -> list[Hit]:
    """Cross-reference such as 'item:1A' or 'note:7': best match for the query inside that section."""
    kind, _, value = ref.partition(":")
    where = {"item": value} if kind == "item" else {"note": value}
    return store.search(query, k=1, where=where)


def section_filter(query: str) -> Optional[dict]:
    """'Item 1A ...' or 'Note 7 ...' in the query restricts the search to that section."""
    if m := re.search(r"\bItem (\d+[A-C]?)\b", query, re.IGNORECASE):
        return {"item": m.group(1).upper()}
    if m := re.search(r"\bNote (\d+)\b", query, re.IGNORECASE):
        return {"note": m.group(1)}
    return None


def document_qa(query: str) -> str:
    """Search Apple's FY2025 10-K. Returns relevant text and tables, each with a citation."""
    try:
        store = get_store()
    except FileNotFoundError as exc:
        return f"ERROR: {exc}"

    hits = store.search(query, k=TOP_K, where=section_filter(query))
    hits = [h for h in hits if h.distance <= MAX_DISTANCE]
    if not hits:
        return "NO_RESULTS: nothing relevant was found in the 10-K for this query."

    # 1. the hits, 2. their linked chunks (text <-> table), 3. chunks their cross-references point to
    wanted: dict[str, None] = {h.id: None for h in hits}
    for h in hits:
        for linked in h.split("linked_ids"):
            wanted.setdefault(linked)
    for h in hits:
        for ref in h.split("refs")[:2]:
            for extra in _follow_reference(store, ref, query):
                wanted.setdefault(extra.id)
    ids = list(wanted)[:MAX_CHUNKS]

    chunks = sorted(store.get(ids), key=lambda c: c.meta["seq"])      # document order
    logger.info("document_qa %r -> %s", query, [c.id for c in chunks])
    parts = []
    for c in chunks:
        body = store.tables[c.id] if c.meta["type"] == "table" else c.text.split("\n\n", 1)[-1]
        parts.append(f"{c.meta['citation']}\n{body}")
    return "\n\n---\n\n".join(parts)
