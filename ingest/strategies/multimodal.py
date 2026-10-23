"""Task 3: hierarchical text + table chunks (full + row-level) + page figure descriptions + linked footnotes."""

import re

from langchain_core.documents import Document

from app.config import DOCUMENTS, LARGE_TABLE_ROWS
from ingest.strategies import hierarchical as h
from ingest.vision import describe_pages, is_decorative

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_FOOTNOTE_MARKER = re.compile(r"^\s*([0-9]{1,2}|[⁰¹²³⁴⁵⁶⁷⁸⁹]+|[*†‡§])[\s.)]")
_NOTES_HEADING = re.compile(r"^\s*(end\s*notes|foot\s*notes|notes)\s*$", re.I)
MAX_FOOTNOTE_LINKS = 3


def footnote_marker(text: str) -> str | None:
    m = _FOOTNOTE_MARKER.match(text)
    return m.group(1).translate(_SUPERSCRIPTS) if m else None


def references_marker(text: str, marker: str) -> bool:
    m = re.escape(marker)
    if marker.isdigit():
        sup = re.escape(marker.translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")))
        patterns = [
            rf"[.,;:)\]%”\"’]\s?{m}(?!\d)(?![.,]\d)",  # "ENERGY STAR. 18", "STAR.18"
            rf"[a-z]\s{m}\s*$",                       # heading "Smarter chemistry 5"
            sup,
            rf"\[{m}\]",
        ]
    else:
        patterns = [rf"\S{m}"]
    return any(re.search(p, text, re.M) for p in patterns)


def split_footnotes(elements: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate footnotes, including endnote sections that Docling tags as list items."""
    body, notes = [], []
    in_notes = False
    for el in elements:
        if el["type"] == "heading":
            in_notes = bool(_NOTES_HEADING.match(el["text"]))
        if el["text"].lstrip().startswith("©"):
            continue  # copyright boilerplate
        if el["type"] == "footnote" or (in_notes and el["type"] in ("list", "paragraph")):
            notes.append(el)
        else:
            body.append(el)
    # Unnumbered note lines (e.g. a definition under note 18) belong to the previous note
    merged: list[dict] = []
    for el in notes:
        if merged and footnote_marker(el["text"]) is None and el["type"] != "footnote":
            merged[-1] = {**merged[-1], "text": merged[-1]["text"] + "\n" + el["text"]}
        else:
            merged.append(el)
    return body, merged


def row_chunks(doc_id: str, parent: h.Parent, el: dict) -> list[Document]:
    rows = el.get("rows") or []
    if len(rows) - 1 <= LARGE_TABLE_ROWS:
        return []
    header, body = rows[0], rows[1:]
    cap = el.get("caption") or "Table"
    out = []
    for row in body:
        label = row[0].strip()
        cells = [
            f"{label} — {col}: {val}"
            for col, val in zip(header[1:], row[1:])
            if str(val).strip() and str(val).strip().lower() not in ("nan", "none")
        ]
        if cells:
            text = f"[Section: {parent.section_path}]\n{cap}\n" + "; ".join(cells)
            out.append(h._doc(doc_id, text, el["page"], parent.section_path, "table", parent.id))
    return out


def build(parsed: dict, doc_id: str, strategy: str = "multimodal") -> tuple[list[Document], list[dict]]:
    page_descriptions = describe_pages(doc_id, DOCUMENTS[doc_id]["title"], parsed)

    body, footnotes = split_footnotes(h.prepare_elements(parsed, doc_id))
    # Pictures survive only through their caption (icon call-outs); visuals come from page descriptions
    elements = [e for e in body if e["type"] != "picture" or e.get("caption")]

    parents = h.make_parents(doc_id, strategy, h.split_sections(elements))
    chunks: list[Document] = []
    for p in parents:
        chunks.extend(h.text_children(doc_id, p, skip_types=("table",)))
        for el in p.elements:
            if el["type"] == "table":
                chunks.extend(h.table_chunks(doc_id, p, el))
                chunks.extend(row_chunks(doc_id, p, el))

    # Figures: one chunk per visual page, attached to the first parent on that page
    for page_no, desc in sorted(page_descriptions.items()):
        parent = next((p for p in parents if page_no in p.pages), None)
        section = parent.section_path if parent else "Figures"
        text = f"[Figures, p. {page_no}]\n{desc}"
        if parent:
            parent.extra.append(text)
        chunks.append(h._doc(doc_id, f"[Section: {section}]\n{text}", page_no, section, "figure", parent.id if parent else None))

    # Footnotes: attach to every parent whose text references the marker (same page first)
    by_id = {p.id: p for p in parents}
    body_chunks = [c for c in chunks if c.metadata["content_type"] != "footnote"]
    for fn in footnotes:
        marker = footnote_marker(fn["text"])
        refs = [c for c in body_chunks if marker and references_marker(c.page_content, marker)]
        refs.sort(key=lambda c: c.metadata["page"] != fn["page"])
        linked: list[h.Parent] = []
        for c in refs:
            parent = by_id.get(c.metadata["parent_id"])
            if parent and parent not in linked:
                linked.append(parent)
            if len(linked) >= MAX_FOOTNOTE_LINKS:
                break
        for parent in linked:
            parent.extra.append(f"Footnote {marker}: {fn['text']}")
        home = linked[0] if linked else None
        section = home.section_path if home else "Footnotes"
        chunks.append(
            h._doc(doc_id, f"[Section: {section}]\nFootnote: {fn['text']}", fn["page"], section, "footnote", home.id if home else None)
        )

    return h.finalize(chunks), [p.row() for p in parents]
