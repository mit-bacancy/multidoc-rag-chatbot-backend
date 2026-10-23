"""Task 2: section-aware parent/child chunking.

sections (from heading tree) -> parents (<= PARENT_MAX_TOKENS, split at element boundaries)
-> children (~CHILD_TARGET_TOKENS, never crossing a parent/section), prefixed with the section path.
Tables become their own chunks.
"""

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHILD_TARGET_TOKENS, DOCUMENTS, PARENT_MAX_TOKENS
from app.embeddings import count_tokens

_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+\S")

_child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_TARGET_TOKENS,
    chunk_overlap=40,
    length_function=count_tokens,
)


@dataclass
class Parent:
    id: str
    section_path: str
    elements: list[dict] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)  # e.g. linked footnotes appended later

    @property
    def pages(self) -> list[int]:
        return [e["page"] for e in self.elements if e["page"] is not None]

    def text(self) -> str:
        """Parent text with a [p. N] marker wherever the page changes, so citations stay exact."""
        parts, page = [], None
        for e in self.elements:
            if e["page"] is not None and e["page"] != page:
                page = e["page"]
                parts.append(f"[p. {page}]")
            parts.append(render(e))
        return "\n\n".join(parts + self.extra)

    def row(self) -> dict:
        pages = self.pages
        return {
            "id": self.id,
            "section_path": self.section_path,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "text": f"[Section: {self.section_path}]\n{self.text()}",
        }


def render(el: dict) -> str:
    """Text of an element as it should appear inside a parent / child."""
    if el["type"] == "table":
        cap = f"Table: {el['caption']}\n" if el.get("caption") else ""
        return cap + (el.get("markdown") or el["text"])
    if el["type"] == "picture":
        cap = el.get("caption") or ""
        desc = el.get("description") or ""
        return f"[Figure, p. {el['page']}] {cap}\n{desc}".strip()
    if el["type"] == "list":
        return f"- {el['text']}"
    return el["text"]


_CAPTION_LIKE = re.compile(r"^(FIGURE|TABLE)\s+[A-Z]?\d|^Sources?:")
_BOX = re.compile(r"^BOX\s+([A-Z]?\d+(?:\.\d+)*)\s*(.*?)\s*(\(continued\))?\s*$", re.I)
_CHAPTER_LABEL = re.compile(r"^CHAPTER\s+\d+$", re.I)
MAX_LEVEL = 4


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def size_level(el: dict, body: float | None) -> int:
    """Docling reports flat levels, so rank headings by font size relative to body text."""
    m = _NUMBERED.match(el["text"])
    size = el.get("font_size")
    if not size or not body:
        return min(m.group(1).count(".") + 1, MAX_LEVEL) if m else 2
    ratio = size / body
    if ratio >= 2.2:
        return 1
    if ratio >= 1.4:
        return 2
    if ratio >= 1.12:
        return 3
    return MAX_LEVEL


def outline_chapters(parsed: dict) -> list[tuple[int, str, int]]:
    """Top-level PDF bookmarks as (page, title, level). Entries marked "--" (e.g. GEP regions)
    are sub-sections of the preceding chapter; boxes and contents pages are skipped."""
    out = []
    for lvl, title, page in parsed.get("outline") or []:
        if lvl != 1 or page < 1 or re.match(r"^(box\b|.*contents)", title, re.I):
            continue
        sub = "--" in title
        title = re.sub(r"\s*--\s*", " ", title).strip()
        title = re.sub(r"^Ch\.\s*", "Chapter ", title)
        out.append((page, title, 1 if sub else 0))
    return sorted(out, key=lambda c: c[0])


def prepare_elements(parsed: dict, doc_id: str) -> list[dict]:
    """Assign heading levels, demote caption-like headings, collapse repeated box headers,
    and (if the document opts in) insert the PDF outline's chapters as level-0 headings."""
    body = parsed.get("body_font_size")
    chapters = outline_chapters(parsed) if DOCUMENTS[doc_id].get("use_outline") else []
    chapter_keys = [_norm(t) for _, t, _ in chapters]
    pending = list(chapters)

    out: list[dict] = []
    open_box = None
    for el in parsed["elements"]:
        while pending and el["page"] is not None and el["page"] >= pending[0][0]:
            page, title, level = pending.pop(0)
            out.append({"type": "heading", "text": title, "page": page, "level": level, "bbox": None})
            open_box = None
        if el["type"] != "heading":
            out.append(el)
            continue

        text = el["text"].strip()
        if _CAPTION_LIKE.match(text) or (el.get("font_size") and body and el["font_size"] < body * 0.7):
            out.append({**el, "type": "paragraph"})  # figure/table titles, panel labels, sources
            continue
        if chapters and (_CHAPTER_LABEL.match(text) or any(_norm(text) in k for k in chapter_keys if len(_norm(text)) > 8)):
            continue  # chapter title pages: the outline already supplies the chapter heading
        box = _BOX.match(text)
        if box:
            if box.group(1) == open_box:
                continue  # "BOX 1.1 ... (continued)" repeated on every page
            open_box = box.group(1)
            out.append({**el, "text": f"Box {box.group(1)} {box.group(2)}".strip(), "level": 2})
            continue
        level = size_level(el, body)
        if chapters and level == 1:
            level = 0  # front-matter titles (Foreword, Abbreviations) are chapter-level, not nested
        if level <= 2:
            open_box = None
        out.append({**el, "level": level})
    return out


def split_sections(elements: list[dict]) -> list[tuple[str, list[dict]]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[dict]]] = []
    current: list[dict] = []

    def path() -> str:
        return " > ".join(t for _, t in stack) or "Front matter"

    for el in elements:
        if el["type"] == "heading":
            if current:
                sections.append((path(), current))
                current = []
            lvl = el["level"]
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, el["text"][:120]))
        else:
            current.append(el)
    if current:
        sections.append((path(), current))
    return sections


def make_parents(doc_id: str, strategy: str, sections) -> list[Parent]:
    parents: list[Parent] = []
    for section_path, elements in sections:
        parent = None
        size = 0
        for el in elements:
            cost = count_tokens(render(el))
            if parent is None or (size + cost > PARENT_MAX_TOKENS and parent.elements):
                parent = Parent(id=f"{doc_id}:{strategy}:{len(parents)}", section_path=section_path)
                parents.append(parent)
                size = 0
            parent.elements.append(el)
            size += cost
    return parents


def _doc(doc_id: str, text: str, page, section_path: str, content_type: str, parent_id) -> Document:
    return Document(
        page_content=text,
        metadata={
            "doc_id": doc_id,
            "page": page,
            "section_path": section_path,
            "content_type": content_type,
            "parent_id": parent_id,
        },
    )


def text_children(doc_id: str, parent: Parent, skip_types=("table", "picture")) -> list[Document]:
    """~CHILD_TARGET_TOKENS chunks of the parent's prose, grouped at element boundaries."""
    prefix = f"[Section: {parent.section_path}]\n"
    out: list[Document] = []
    buf: list[str] = []
    buf_page = None
    size = 0

    def flush():
        nonlocal buf, size, buf_page
        if buf:
            out.append(_doc(doc_id, prefix + "\n\n".join(buf), buf_page, parent.section_path, "text", parent.id))
        buf, size, buf_page = [], 0, None

    for el in parent.elements:
        if el["type"] in skip_types:
            continue
        body = render(el)
        cost = count_tokens(body)
        if cost > CHILD_TARGET_TOKENS:
            flush()
            for piece in _child_splitter.split_text(body):
                out.append(_doc(doc_id, prefix + piece, el["page"], parent.section_path, "text", parent.id))
            continue
        if size + cost > CHILD_TARGET_TOKENS:
            flush()
        if buf_page is None:
            buf_page = el["page"]
        buf.append(body)
        size += cost
    flush()
    return out


TABLE_CHUNK_TOKENS = 450  # BGE truncates input at 512 tokens


def table_chunks(doc_id: str, parent: Parent, el: dict) -> list[Document]:
    """Whole table as one chunk, or row groups that repeat the caption and header when it is too large."""
    prefix = f"[Section: {parent.section_path}]\n"
    full = render(el)
    if count_tokens(prefix + full) <= TABLE_CHUNK_TOKENS:
        return [_doc(doc_id, prefix + full, el["page"], parent.section_path, "table", parent.id)]

    lines = (el.get("markdown") or el["text"]).splitlines()
    head, rows = lines[:2], lines[2:]  # markdown header + separator
    cap = f"Table: {el['caption']}\n" if el.get("caption") else ""
    base = prefix + cap + "\n".join(head)
    budget = TABLE_CHUNK_TOKENS - count_tokens(base)
    out, group, size = [], [], 0
    for row in rows:
        cost = count_tokens(row)
        if group and size + cost > budget:
            out.append(_doc(doc_id, base + "\n" + "\n".join(group), el["page"], parent.section_path, "table", parent.id))
            group, size = [], 0
        group.append(row)
        size += cost
    if group:
        out.append(_doc(doc_id, base + "\n" + "\n".join(group), el["page"], parent.section_path, "table", parent.id))
    return out


def finalize(chunks: list[Document]) -> list[Document]:
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


def build(parsed: dict, doc_id: str, strategy: str = "hierarchical") -> tuple[list[Document], list[dict]]:
    # Without vision captions an uncaptioned picture carries no text
    elements = [e for e in prepare_elements(parsed, doc_id) if e["type"] != "picture" or e.get("caption")]
    parents = make_parents(doc_id, strategy, split_sections(elements))
    chunks: list[Document] = []
    for p in parents:
        chunks.extend(text_children(doc_id, p))
        for el in p.elements:
            if el["type"] == "table":
                chunks.extend(table_chunks(doc_id, p, el))
    return finalize(chunks), [p.row() for p in parents]
