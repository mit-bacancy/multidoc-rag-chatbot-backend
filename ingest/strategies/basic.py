"""Task 1 baseline: plain text joined page by page, fixed-size chunks, page metadata only."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import BASIC_CHUNK_OVERLAP, BASIC_CHUNK_SIZE


def build(parsed: dict, doc_id: str) -> tuple[list[Document], list[dict]]:
    pages: dict[int, list[str]] = {}
    for el in parsed["elements"]:
        if el["page"] is not None and el["text"]:
            pages.setdefault(el["page"], []).append(el["text"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASIC_CHUNK_SIZE, chunk_overlap=BASIC_CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    for page in sorted(pages):
        for piece in splitter.split_text("\n\n".join(pages[page])):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": doc_id,
                        "page": page,
                        "section_path": None,
                        "content_type": "text",
                        "parent_id": None,
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks, []
