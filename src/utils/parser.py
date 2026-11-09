"""Parse Apple's 10-K HTML into an ordered list of blocks (paragraphs and tables).

Every block remembers where it sits in the filing: Item, Note, heading path and page.
Page numbers come from the footer line "Apple Inc. | 2025 Form 10-K | 24", which closes
each page, so every block before a footer belongs to that page.
"""
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

FOOTER_RE = re.compile(r"^Apple Inc\. \| \d{4} Form 10-K \| (\d+)$")
ITEM_RE = re.compile(r"^Item\s+(\d+[A-C]?)\.\s+\S")
NOTE_RE = re.compile(r"^Note\s+(\d+)\s*[–-]\s*(.+)$")
PART_RE = re.compile(r"^PART\s+[IVX]+$", re.IGNORECASE)
MAX_HEADING_CHARS = 400          # bold lines longer than this are treated as text
MAX_SUBHEADING_CHARS = 80        # short italic lines ("iPhone", "Products Gross Margin")
SKIP_TAGS = {"hr", "br", "script", "style", "head"}


@dataclass
class Block:
    """One paragraph or one table, with its position in the filing."""
    kind: str                                   # "text" or "table"
    text: str                                   # paragraph text, or the sentence above a table
    item: str = ""                              # "1A", "7", "8" ... ("" before Item 1)
    note: str = ""                              # note number inside Item 8
    note_title: str = ""
    heading_path: list[str] = field(default_factory=list)   # [heading, sub-heading]
    page: int = 0
    rows: list[list[str]] = field(default_factory=list)     # tables only: [label, value, value...]


def clean(text: str) -> str:
    """Collapse whitespace and non-breaking spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _hidden(tag: Tag) -> bool:
    return "display:none" in (tag.get("style") or "").replace(" ", "")


def _walk(tag: Tag) -> Iterator[Tag]:
    """Yield tables and leaf paragraphs (elements with no div/table inside) in document order."""
    for child in tag.children:
        if not isinstance(child, Tag) or child.name in SKIP_TAGS or _hidden(child):
            continue
        if child.name == "table":
            yield child
        elif child.find(["div", "table"]) is not None:
            yield from _walk(child)              # wrapper (div, ix:nonnumeric ...)
        else:
            yield child                          # leaf paragraph


def _text_spans(tag: Tag) -> list[Tag]:
    return [s for s in tag.find_all("span") if s.get_text(strip=True) and not s.find("span")]


def _all_spans_have(tag: Tag, css: str) -> bool:
    spans = _text_spans(tag)
    return bool(spans) and all(css in (s.get("style") or "").replace(" ", "") for s in spans)


def table_rows(table: Tag) -> list[list[str]]:
    """Return table rows as [label, value, value, ...].

    EDGAR puts '$' and '%' in their own cells and leaves empty spacer cells, so values are
    merged back together ("7" + "%" -> "7%") and empty cells are dropped.
    """
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        label = clean(cells[0].get_text(" "))
        values: list[str] = []
        for cell in cells[1:]:
            text = clean(cell.get_text(" "))
            if not text or text == "$":
                continue
            if text in ("%", ")", "%)", ")%") and values:
                values[-1] += text
            else:
                values.append(text)
        if label or values:
            rows.append([label] + values)
    return rows


def parse_10k(path: Path) -> list[Block]:
    """Parse the 10-K HTML file into blocks with section and page context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)   # the iXBRL file is XHTML
        soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="ignore"), "lxml")
    blocks: list[Block] = []
    item = note = note_title = ""
    headings: list[str] = []
    page_start = 0
    last_page = 0

    for el in _walk(soup.body):
        if el.name == "table":
            rows = table_rows(el)
            if len(rows) < 2:                    # layout table with one row: treat as text
                text = clean(" ".join(" ".join(r) for r in rows))
                if text:
                    blocks.append(Block("text", text, item, note, note_title, list(headings)))
                continue
            prev = blocks[-1] if blocks else None
            if prev and prev.kind == "table" and prev.heading_path == headings:
                prev.rows += rows                # two adjacent tables under one heading: one table
                continue
            intro = ""
            if prev and prev.kind == "text" and prev.text.endswith(":"):
                intro = prev.text                # e.g. "... were as follows (dollars in millions):"
            blocks.append(Block("table", intro, item, note, note_title, list(headings), rows=rows))
            continue

        text = clean(el.get_text(""))
        if not text:
            continue

        footer = FOOTER_RE.match(text)
        if footer:                               # a footer closes the page
            last_page = int(footer.group(1))
            for b in blocks[page_start:]:
                b.page = last_page
            page_start = len(blocks)
            continue
        if PART_RE.match(text):
            continue

        bold = _all_spans_have(el, "font-weight:700") and len(text) <= MAX_HEADING_CHARS
        if ITEM_RE.match(text) and (bold or len(text) < 100):
            item, note, note_title, headings = ITEM_RE.match(text).group(1), "", "", []
            continue
        if note_match := NOTE_RE.match(text):
            if bold or len(text) < 100:
                note, note_title, headings = note_match.group(1), note_match.group(2), []
                continue
        italic = _all_spans_have(el, "font-style:italic")
        if bold:                                 # a heading, not a sentence
            headings = headings[:1] + [text] if (italic and headings) else [text]
            continue
        if italic and len(text) <= MAX_SUBHEADING_CHARS and not text.endswith("."):
            headings = headings[:1] + [text]     # short italic line = sub-heading
            continue

        blocks.append(Block("text", text, item, note, note_title, list(headings)))

    for b in blocks[page_start:]:                # anything after the last footer
        b.page = last_page + 1
    return blocks
