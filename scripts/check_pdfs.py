"""Sanity-check the PDFs: page count, text layer, pages with images / tables.

    python -m scripts.check_pdfs
"""

import pymupdf as fitz

from app.config import DOCUMENTS, PDF_DIR


def check(doc_id: str, file: str) -> None:
    path = PDF_DIR / file
    if not path.exists():
        print(f"{doc_id:12} MISSING  {path}")
        return
    pdf = fitz.open(path)
    no_text = img_pages = table_pages = 0
    for page in pdf:
        if len(page.get_text().strip()) < 30:
            no_text += 1
        if page.get_images():
            img_pages += 1
        try:
            if page.find_tables().tables:
                table_pages += 1
        except Exception:
            pass
    verdict = "OK" if no_text < 0.2 * len(pdf) else "NEEDS OCR (--ocr)"
    print(
        f"{doc_id:12} {len(pdf):4} pages | no-text pages: {no_text:3} | "
        f"pages w/ images: {img_pages:3} | pages w/ tables: {table_pages:3} | {verdict}"
    )


def main() -> None:
    for doc_id, doc in DOCUMENTS.items():
        check(doc_id, doc["file"])


if __name__ == "__main__":
    main()
