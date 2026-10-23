import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
PARSED_DIR = BASE_DIR / "data" / "parsed"

# Supabase hands out postgresql:// URLs; psycopg 3 needs the +psycopg driver prefix
SUPABASE_DB_URL = re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", os.getenv("SUPABASE_DB_URL", ""))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
VISION_MODEL = os.getenv("VISION_MODEL", LLM_MODEL)

# Pinned: ingest and query must use the same model (768-dim vectors)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
# bge-reranker-base took 12s+/query on CPU; MiniLM-L6 keeps the right page in the top 6
# for all sample questions at ~1.2s
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
PARSER_VERSION = "docling-v1"

# Single registry every module reads from
DOCUMENTS = {
    "gpo": {
        "title": "GPO 2018 Annual Report",
        "file": "GPO-Anual-report.pdf",
        "strategy": "hierarchical",
        "baseline_strategy": "basic",
    },
    "worldbank": {
        "title": "Global Economic Prospects, June 2026",
        "file": "GEP-Jun-2026.pdf",
        "strategy": "hierarchical",
        "use_outline": True,  # PDF bookmarks give reliable chapter boundaries
    },
    "apple_imac": {
        "title": "iMac Product Environmental Report",
        "file": "iMac_PER_Oct2024.pdf",
        "strategy": "multimodal",
    },
}

STRATEGIES = ("basic", "hierarchical", "multimodal")

# Chunking
BASIC_CHUNK_SIZE = 1000
BASIC_CHUNK_OVERLAP = 150
PARENT_MAX_TOKENS = 2000
CHILD_TARGET_TOKENS = 350
LARGE_TABLE_ROWS = 8  # tables with more rows also get row-level chunks
EMBED_BATCH_SIZE = 64

# Retrieval
VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RERANK_CANDIDATES = 20  # RRF results passed to the cross-encoder
RERANK_TOP_N = 6
CONTEXT_MAX_TOKENS = 7000


def collection_name(doc_id: str, strategy: str) -> str:
    return f"{doc_id}__{strategy}"


def doc_strategies(doc_id: str) -> list[str]:
    doc = DOCUMENTS[doc_id]
    strategies = [doc["strategy"]]
    if doc.get("baseline_strategy"):
        strategies.append(doc["baseline_strategy"])
    return strategies
