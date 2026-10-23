"""Render visual pages with PyMuPDF and describe them with Gemini vision (cached on disk).

Whole pages rather than picture crops: many charts are vector drawings that Docling does not
report as pictures, and a page render also keeps icon call-outs next to their numbers.
"""

import json
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import PARSED_DIR
from app.llm import describe_image
from ingest.parse import pdf_path

logger = logging.getLogger(__name__)

PROMPT = (
    "This is page {page} of the report \"{title}\".\n"
    "Describe every chart, graph, infographic, diagram and icon call-out on the page factually, "
    "so the description can answer questions without the image. For each one give: its title, the "
    "chart type, every axis label and unit, every legend entry, and every number, percentage and "
    "year that is visible, paired with exactly what it refers to. Keep footnote markers (small "
    "numbers next to a claim) as [n]. Skip ordinary body paragraphs, headers and footers. "
    "Do not interpret or speculate beyond what is shown. Plain text, no preamble. "
    "If the page has no visual elements, reply with NONE."
)

MIN_SIDE_PT = 80          # smaller and uncaptioned -> decoration (logo/icon)
MIN_AREA_RATIO = 0.02
VISUAL_DRAWINGS_MIN = 25  # vector paths on a page that suggest a chart / infographic
RENDER_ZOOM = 2.0


def cache_path(doc_id: str):
    return PARSED_DIR / f"{doc_id}.pages.json"


def is_decorative(el: dict, page_size: list[float]) -> bool:
    """Small picture (logo/icon). Captioned icons keep their caption text but are not sent to vision."""
    if not el.get("bbox"):
        return True
    l, t, r, b = el["bbox"]
    w, h = abs(r - l), abs(b - t)
    pw, ph = page_size
    return min(w, h) < MIN_SIDE_PT or (w * h) / (pw * ph) < MIN_AREA_RATIO


def visual_pages(parsed: dict, pdf) -> list[int]:
    pages = {
        el["page"] for el in parsed["elements"]
        if el["type"] == "picture" and not is_decorative(el, parsed["page_sizes"][str(el["page"])])
    }
    pages |= {p.number + 1 for p in pdf if len(p.get_drawings()) >= VISUAL_DRAWINGS_MIN}
    return sorted(pages)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def _describe(png: bytes, prompt: str) -> str:
    return describe_image(png, prompt)


def describe_pages(doc_id: str, title: str, parsed: dict) -> dict[int, str]:
    """Return {page: description} for pages with charts / figures."""
    import pymupdf

    cache_file = cache_path(doc_id)
    cache: dict[str, str] = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    pdf = pymupdf.open(pdf_path(doc_id))
    pages = visual_pages(parsed, pdf)
    logger.info("Vision: %d visual pages %s (%d cached)", len(pages), pages, sum(str(p) in cache for p in pages))

    for page_no in pages:
        if str(page_no) in cache:
            continue
        png = pdf[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)).tobytes("png")
        try:
            cache[str(page_no)] = _describe(png, PROMPT.format(page=page_no, title=title)).strip()
        except Exception:
            logger.exception("Vision failed for page %d", page_no)
            continue
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1))  # save as we go

    return {
        int(p): d for p, d in cache.items()
        if int(p) in pages and d and d.strip().upper() != "NONE"
    }
