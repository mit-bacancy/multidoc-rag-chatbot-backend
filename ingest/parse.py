"""Docling -> ordered, page-tagged elements, cached as JSON in data/parsed/<doc_id>.json.

Element shape:
    {"type": "heading|paragraph|list|table|picture|footnote",
     "text": str, "page": int, "level": int|None,
     "bbox": [l, t, r, b] (top-left origin, PDF points) | None,
     "caption": str|None,                       # tables / pictures
     "markdown": str|None, "rows": [[str]]|None # tables
    }
"""

import json
import logging
from collections import Counter
from pathlib import Path

from app.config import DOCUMENTS, PARSED_DIR, PARSER_VERSION, PDF_DIR

logger = logging.getLogger(__name__)

SKIP_LABELS = {"page_header", "page_footer"}
TYPE_BY_LABEL = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list",
    "table": "table",
    "document_index": "table",
    "picture": "picture",
    "chart": "picture",
    "footnote": "footnote",
    "formula": "paragraph",
    "code": "paragraph",
    "key_value_region": "paragraph",
}


def parsed_path(doc_id: str) -> Path:
    return PARSED_DIR / f"{doc_id}.json"


def pdf_path(doc_id: str) -> Path:
    return PDF_DIR / DOCUMENTS[doc_id]["file"]


def _converter(do_ocr: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _bbox(item, doc) -> tuple[int | None, list[float] | None]:
    if not item.prov:
        return None, None
    prov = item.prov[0]
    page = prov.page_no
    height = doc.pages[page].size.height
    bb = prov.bbox.to_top_left_origin(page_height=height)
    return page, [bb.l, bb.t, bb.r, bb.b]


def _table_rows(table, doc) -> list[list[str]]:
    try:
        df = table.export_to_dataframe(doc=doc)
    except TypeError:
        df = table.export_to_dataframe()
    header = [str(c) for c in df.columns]
    return [header] + [[str(v) for v in row] for row in df.itertuples(index=False)]


def convert(doc_id: str, do_ocr: bool = False) -> dict:
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem

    path = pdf_path(doc_id)
    logger.info("Docling: converting %s (ocr=%s)", path.name, do_ocr)
    doc = _converter(do_ocr).convert(str(path)).document

    # Captions are emitted both standalone and as children of tables/pictures; keep them on the parent
    caption_refs = {
        cap.cref
        for item, _ in doc.iterate_items()
        if isinstance(item, (TableItem, PictureItem))
        for cap in item.captions
    }

    elements: list[dict] = []
    for item, _depth in doc.iterate_items():
        label = str(getattr(item.label, "value", item.label))
        if label in SKIP_LABELS or item.self_ref in caption_refs:
            continue
        etype = TYPE_BY_LABEL.get(label)
        if label == "caption":
            etype = "paragraph"  # orphan caption
        if etype is None:
            continue
        page, bbox = _bbox(item, doc)
        el = {"type": etype, "page": page, "bbox": bbox, "level": None, "caption": None,
              "markdown": None, "rows": None, "text": ""}

        if isinstance(item, TableItem):
            el["caption"] = item.caption_text(doc) or None
            el["markdown"] = item.export_to_markdown(doc=doc)
            el["rows"] = _table_rows(item, doc)
            el["text"] = el["markdown"]
        elif isinstance(item, PictureItem):
            el["caption"] = item.caption_text(doc) or None
            el["text"] = el["caption"] or ""
        else:
            el["text"] = (getattr(item, "text", "") or "").strip()
            if not el["text"]:
                continue
            if etype == "heading":
                el["level"] = item.level if isinstance(item, SectionHeaderItem) else 0
        elements.append(el)

    page_sizes = {str(no): [p.size.width, p.size.height] for no, p in doc.pages.items()}
    return {
        "doc_id": doc_id,
        "parser_version": PARSER_VERSION,
        "page_count": len(doc.pages),
        "page_sizes": page_sizes,
        "elements": elements,
    }


def enrich(parsed: dict, doc_id: str) -> bool:
    """Add what Docling lacks: heading font sizes (Docling levels are flat), body font size, PDF outline."""
    if parsed.get("enriched"):
        return False
    import pymupdf

    pdf = pymupdf.open(pdf_path(doc_id))
    parsed["outline"] = [[lvl, " ".join(title.split()), page] for lvl, title, page in pdf.get_toc()]

    chars_by_size: Counter = Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    chars_by_size[round(span["size"], 1)] += len(span["text"].strip())
    parsed["body_font_size"] = chars_by_size.most_common(1)[0][0] if chars_by_size else None

    for el in parsed["elements"]:
        if el["type"] != "heading" or not el["bbox"]:
            continue
        clip = pymupdf.Rect(el["bbox"]) + (-2, -2, 2, 2)
        spans = [
            s
            for b in pdf[el["page"] - 1].get_text("dict", clip=clip)["blocks"]
            for line in b.get("lines", [])
            for s in line["spans"]
            if s["text"].strip()
        ]
        if spans:
            top = max(spans, key=lambda s: s["size"])
            el["font_size"] = round(top["size"], 1)
            el["bold"] = bool(top["flags"] & 16) or "bold" in top["font"].lower()
    parsed["enriched"] = True
    return True


def parse(doc_id: str, force: bool = False, do_ocr: bool = False) -> dict:
    """Parse once and cache; strategies read the cache."""
    out = parsed_path(doc_id)
    parsed = None
    if out.exists() and not force:
        cached = json.loads(out.read_text())
        if cached.get("parser_version") == PARSER_VERSION:
            parsed = cached
    if parsed is None:
        parsed = convert(doc_id, do_ocr=do_ocr)
        counts = Counter(el["type"] for el in parsed["elements"])
        logger.info("Parsed %s: %d pages, %s", doc_id, parsed["page_count"], dict(counts))
    elif parsed.get("enriched"):
        return parsed
    enrich(parsed, doc_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parsed, ensure_ascii=False))
    return parsed
