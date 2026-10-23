"""Development-time ingestion.

    python -m ingest.run_ingest --doc gpo|worldbank|apple_imac|all [--strategy X] [--rebuild] [--reparse] [--ocr]
"""

import argparse
import logging
from collections import Counter

from app import store
from app.config import DOCUMENTS, EMBED_BATCH_SIZE, PARSER_VERSION, STRATEGIES, doc_strategies
from ingest.parse import parse, pdf_path
from ingest.strategies import basic, hierarchical, multimodal

BUILDERS = {
    "basic": lambda parsed, doc_id: basic.build(parsed, doc_id),
    "hierarchical": lambda parsed, doc_id: hierarchical.build(parsed, doc_id),
    "multimodal": lambda parsed, doc_id: multimodal.build(parsed, doc_id),
}

logger = logging.getLogger("ingest")


def ingest(doc_id: str, strategy: str, rebuild: bool, reparse: bool, ocr: bool) -> None:
    if not pdf_path(doc_id).exists():
        logger.error("Missing PDF: %s", pdf_path(doc_id))
        return
    runs = store.latest_runs()
    if (doc_id, strategy) in runs and not rebuild:
        logger.info("Skip %s/%s: already ingested (use --rebuild)", doc_id, strategy)
        return

    parsed = parse(doc_id, force=reparse, do_ocr=ocr)
    chunks, parents = BUILDERS[strategy](parsed, doc_id)

    store.reset_collection(doc_id, strategy)
    store.save_parents(doc_id, strategy, parents)
    vs = store.get_vectorstore(doc_id, strategy)
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        vs.add_documents(chunks[i : i + EMBED_BATCH_SIZE])
        logger.info("  embedded %d/%d", min(i + EMBED_BATCH_SIZE, len(chunks)), len(chunks))

    store.record_run(doc_id, strategy, len(chunks), parsed["page_count"], PARSER_VERSION)
    by_type = Counter(c.metadata["content_type"] for c in chunks)
    logger.info("Done %s/%s: %d chunks %s, %d parents", doc_id, strategy, len(chunks), dict(by_type), len(parents))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, choices=[*DOCUMENTS, "all"])
    ap.add_argument("--strategy", choices=STRATEGIES, help="default: the doc's strategy (+ baseline)")
    ap.add_argument("--rebuild", action="store_true", help="re-ingest even if already recorded")
    ap.add_argument("--reparse", action="store_true", help="ignore the cached Docling parse")
    ap.add_argument("--ocr", action="store_true", help="enable Docling OCR (scanned PDFs)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    store.init_schema()

    docs = list(DOCUMENTS) if args.doc == "all" else [args.doc]
    for doc_id in docs:
        for strategy in [args.strategy] if args.strategy else doc_strategies(doc_id):
            ingest(doc_id, strategy, args.rebuild, args.reparse, args.ocr)
    store.ensure_hnsw_index()


if __name__ == "__main__":
    main()
