"""Hybrid search (vector + BM25, RRF) -> cross-encoder rerank -> parent expansion."""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app import store
from app.config import (
    BM25_TOP_K,
    CONTEXT_MAX_TOKENS,
    RERANK_CANDIDATES,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RRF_K,
    VECTOR_TOP_K,
)
from app.embeddings import count_tokens, get_embeddings

logger = logging.getLogger("uvicorn.error")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


@dataclass
class ContextBlock:
    text: str
    page: int | None
    page_label: str  # page range of the expanded parent, e.g. "4-5"
    section_path: str | None
    content_type: str
    snippet: str


def _page_label(parent: dict, page) -> str:
    start, end = parent.get("page_start"), parent.get("page_end")
    if start is None:
        return str(page or "?")
    return str(start) if start == end else f"{start}-{end}"


class DocRetriever:
    def __init__(self, doc_id: str, strategy: str):
        self.doc_id = doc_id
        self.strategy = strategy
        self.chunks, self.vectors = store.load_all_chunks(doc_id, strategy)
        self.parents = store.load_parents(doc_id, strategy) if strategy != "basic" else {}
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.chunks]) if self.chunks else None
        logger.info("Loaded %s/%s: %d chunks, %d parents", doc_id, strategy, len(self.chunks), len(self.parents))

    def _vector_search(self, query: str, k: int) -> list[Document]:
        if not self.chunks:
            return []
        q = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
        scores = self.vectors @ q  # embeddings are normalized, so dot product = cosine
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top if scores[i] > 0]

    def hybrid_search(self, query: str) -> list[Document]:
        vector_hits = self._vector_search(query, VECTOR_TOP_K)
        bm25_hits = self._bm25_search(query, BM25_TOP_K)

        # Reciprocal rank fusion
        scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, doc in enumerate(hits):
                key = str(doc.id or hash(doc.page_content))
                by_id[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [by_id[k] for k in ranked]

    def rerank(self, query: str, docs: list[Document], top_n: int = RERANK_TOP_N) -> list[Document]:
        if not docs:
            return []
        docs = docs[:RERANK_CANDIDATES]
        scores = get_reranker().predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:top_n]]

    def expand_parents(self, docs: list[Document]) -> list[ContextBlock]:
        """Swap children for their parent sections (deduped) within the token budget."""
        parents = self.parents

        blocks: list[ContextBlock] = []
        seen: set[str] = set()
        budget = CONTEXT_MAX_TOKENS
        for d in docs:
            meta = d.metadata
            parent = parents.get(meta.get("parent_id"))
            key = meta.get("parent_id") if parent else str(d.id)
            if key in seen:
                continue
            body = parent["text"] if parent else d.page_content
            cost = count_tokens(body)
            if cost > budget:
                # Parent too big for what's left: fall back to the child itself
                body = d.page_content
                cost = count_tokens(body)
                if cost > budget:
                    continue
            seen.add(key)
            budget -= cost
            blocks.append(
                ContextBlock(
                    text=body,
                    page=meta.get("page"),
                    page_label=_page_label(parent, meta.get("page")) if parent and body is parent["text"] else str(meta.get("page") or "?"),
                    section_path=meta.get("section_path"),
                    content_type=meta.get("content_type", "text"),
                    snippet=d.page_content[:240],
                )
            )
        return blocks

    def retrieve(self, query: str) -> list[ContextBlock]:
        candidates = self.hybrid_search(query)
        top = self.rerank(query, candidates)
        return self.expand_parents(top)


RETRIEVERS: dict[tuple[str, str], DocRetriever] = {}


def load_retrievers(pairs: list[tuple[str, str]]) -> None:
    # Warm up both models so the first request doesn't pay their initialisation cost
    get_embeddings().embed_query("warm up")
    get_reranker().predict([("warm up", "warm up")])
    for doc_id, strategy in pairs:
        RETRIEVERS[(doc_id, strategy)] = DocRetriever(doc_id, strategy)


def get_retriever(doc_id: str, strategy: str) -> DocRetriever:
    key = (doc_id, strategy)
    if key not in RETRIEVERS:
        RETRIEVERS[key] = DocRetriever(doc_id, strategy)
    return RETRIEVERS[key]
