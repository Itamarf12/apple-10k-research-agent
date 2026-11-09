"""Turn parsed blocks into chunks with metadata and links.

Chunking rules
- Text: paragraphs under the same heading are grouped up to about CHUNK_WORDS words.
  A chunk never crosses an Item, Note or heading boundary.
- Tables: one table = one chunk, never split. Shown to the LLM as a Markdown table.
Every chunk is embedded with a location line in front, e.g.
"[Apple 10-K FY2025 › Item 7 – MD&A › Gross Margin]", so it never loses its context.
"""
import re
from dataclasses import dataclass, field

from config import CHUNK_WORDS, DOC_LABEL
from utils.parser import Block

ITEM_NAMES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Registrant's Common Equity",
    "6": "Reserved", "7": "MD&A", "7A": "Market Risk Disclosures",
    "8": "Financial Statements", "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Foreign Jurisdictions Inspections", "10": "Directors and Corporate Governance",
    "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Related Transactions", "14": "Accountant Fees",
    "15": "Exhibits and Financial Statement Schedules", "16": "Form 10-K Summary",
}

# Apple's reporting entities: five geographic segments and its product categories.
ENTITIES = [
    "Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific",
    "iPhone", "Mac", "iPad", "Wearables, Home and Accessories", "Services", "Products",
]

# Phrases that point to another section, besides "Note N" and "Item N".
PHRASE_REFS = {
    "Risk Factors": "item:1A",
    "Management’s Discussion and Analysis": "item:7",
    "Financial Statements and Supplementary Data": "item:8",
}


@dataclass
class Chunk:
    id: str
    type: str                                    # "text" or "table"
    text: str                                    # what gets embedded (location line + content)
    markdown: str = ""                           # tables only: the full table for the LLM
    seq: int = 0                                 # position in the document
    item: str = ""
    note: str = ""
    heading_path: str = ""
    page: int = 0
    entities: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    linked_ids: list[str] = field(default_factory=list)
    citation: str = ""
    section_key: tuple = ()                      # used for linking text <-> tables


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def item_label(item: str, note: str = "", note_title: str = "") -> str:
    label = f"Item {item} – {ITEM_NAMES.get(item, '')}".strip(" –")
    if note:
        label += f", Note {note}" + (f" – {note_title}" if note_title else "")
    return label


def location_line(item: str, note: str, note_title: str, headings: list[str]) -> str:
    parts = [f"Apple {DOC_LABEL}", item_label(item).replace(", ", " › ")]
    if note:
        parts.append(f"Note {note}" + (f" – {note_title}" if note_title else ""))
    parts += [_short(h, 80) for h in headings]
    return "[" + " › ".join(parts) + "]"


def make_citation(item: str, note: str, note_title: str, headings: list[str], pages: tuple[int, int]) -> str:
    parts = [DOC_LABEL, item_label(item, note, note_title)]
    if headings:
        parts.append(_short(headings[-1], 60))
    first, last = pages
    parts.append(f"p. {first}" if first == last else f"pp. {first}–{last}")
    return "[" + ", ".join(parts) + "]"


def find_entities(text: str) -> list[str]:
    return [e for e in ENTITIES if re.search(rf"\b{re.escape(e)}\b", text)]


def find_refs(text: str, own_item: str, own_note: str) -> list[str]:
    """Cross-references such as 'Note 7', 'Item 1A' or 'under the heading Risk Factors'."""
    refs: list[str] = []
    for n in re.findall(r"\bNote (\d+)\b", text):
        refs.append(f"note:{n}")
    for i in re.findall(r"\bItem (\d+[A-C]?)\b", text):
        refs.append(f"item:{i}")
    for phrase, ref in PHRASE_REFS.items():
        if re.search(rf"[“\"]{re.escape(phrase)}", text):
            refs.append(ref)
    own = {f"item:{own_item}"} | ({f"note:{own_note}"} if own_note else set())
    return [r for r in dict.fromkeys(refs) if r not in own]


def split_long(text: str, max_words: int) -> list[str]:
    """Split one very long paragraph at sentence boundaries."""
    if len(text.split()) <= max_words:
        return [text]
    pieces, current = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def to_markdown(rows: list[list[str]]) -> str:
    """Rows are [label, value, ...]. The first row is the header."""
    if len(rows) > 2 and not rows[0][0] and not rows[1][0] and len(rows[0]) < len(rows[1]):
        rows = [[" ".join(rows[0][1:])] + rows[1][1:]] + rows[2:]     # "Years ended" above the dates
    width = max(len(r) for r in rows)
    padded = [[c.replace("|", "/") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join(lines)


def build_chunks(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []
    group: list[Block] = []                      # paragraphs being merged into one chunk

    def new_id(item: str, note: str, kind: str) -> str:
        where = f"item{item}" + (f"-note{note}" if note else "")
        return f"{where}-{len(chunks):04d}-{'tbl' if kind == 'table' else 'txt'}"

    def flush() -> None:
        if not group:
            return
        first = group[0]
        body = "\n\n".join(b.text for b in group)
        headings = first.heading_path
        chunks.append(Chunk(
            id=new_id(first.item, first.note, "text"), type="text",
            text=location_line(first.item, first.note, first.note_title, headings) + "\n\n" + body,
            seq=len(chunks), item=first.item, note=first.note,
            heading_path=" > ".join(headings), page=first.page,
            entities=find_entities(body), refs=find_refs(body, first.item, first.note),
            citation=make_citation(first.item, first.note, first.note_title, headings,
                                   (first.page, group[-1].page)),
            section_key=(first.item, first.note, headings[0] if headings else ""),
        ))
        group.clear()

    def group_words() -> int:
        return sum(len(b.text.split()) for b in group)

    for block in blocks:
        if not block.item:                       # cover page and table of contents
            continue
        if block.kind == "table":
            flush()
            chunks.append(table_chunk(block, len(chunks), new_id(block.item, block.note, "table")))
            continue
        for piece in split_long(block.text, CHUNK_WORDS):
            part = Block("text", piece, block.item, block.note, block.note_title,
                         block.heading_path, block.page)
            key = lambda b: (b.item, b.note, tuple(b.heading_path))
            if group and (key(group[0]) != key(part) or group_words() + len(piece.split()) > CHUNK_WORDS):
                flush()
            group.append(part)
    flush()
    link_chunks(chunks)
    return chunks


def table_chunk(block: Block, seq: int, chunk_id: str) -> Chunk:
    labels = [r[0] for r in block.rows[1:] if r[0]]
    header = [c for c in block.rows[0][1:] if c]
    intro = block.text or (block.heading_path[-1] if block.heading_path else "Table")
    embed = (location_line(block.item, block.note, block.note_title, block.heading_path)
             + f"\n\nTable: {intro}\nRows: {'; '.join(labels[:30])}"
             + (f"\nColumns: {', '.join(header)}" if header else ""))
    markdown = (block.text + "\n\n" if block.text else "") + to_markdown(block.rows)
    flat = " ".join(" ".join(r) for r in block.rows) + " " + block.text
    return Chunk(
        id=chunk_id, type="table", text=embed, markdown=markdown, seq=seq,
        item=block.item, note=block.note, heading_path=" > ".join(block.heading_path),
        page=block.page, entities=find_entities(flat),
        refs=find_refs(block.text, block.item, block.note),
        citation=make_citation(block.item, block.note, block.note_title, block.heading_path,
                               (block.page, block.page)),
        section_key=(block.item, block.note, block.heading_path[0] if block.heading_path else ""),
    )


def link_chunks(chunks: list[Chunk], per_table: int = 4) -> None:
    """Link each table with the nearest text chunks of the same section, both ways."""
    for table in (c for c in chunks if c.type == "table"):
        same = [c for c in chunks if c.type == "text" and c.section_key == table.section_key]
        same.sort(key=lambda c: abs(c.seq - table.seq))
        for text in same[:per_table]:
            table.linked_ids.append(text.id)
            text.linked_ids.append(table.id)
