"""PGVector collections (one per doc+strategy) plus the parent_chunks / ingest_runs tables."""

import json
from functools import lru_cache

import numpy as np

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import EMBEDDING_DIM, SUPABASE_DB_URL, collection_name
from app.embeddings import get_embeddings

SCHEMA_SQL = """
create extension if not exists vector;

create table if not exists parent_chunks (
    id text primary key,
    doc_id text not null,
    strategy text not null,
    section_path text,
    page_start int,
    page_end int,
    text text not null
);
create index if not exists parent_chunks_doc_idx on parent_chunks (doc_id, strategy);

create table if not exists ingest_runs (
    id bigserial primary key,
    doc_id text not null,
    strategy text not null,
    chunk_count int not null,
    page_count int,
    parser_version text not null,
    created_at timestamptz not null default now()
);
create index if not exists ingest_runs_doc_idx on ingest_runs (doc_id, strategy);
"""


@lru_cache
def get_engine() -> Engine:
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL is not set")
    return create_engine(SUPABASE_DB_URL, pool_pre_ping=True)


def init_schema() -> None:
    with get_engine().begin() as conn:
        for stmt in filter(str.strip, SCHEMA_SQL.split(";")):
            conn.execute(text(stmt))


def ping() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:
        return False


@lru_cache
def get_vectorstore(doc_id: str, strategy: str) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name(doc_id, strategy),
        connection=get_engine(),
        embedding_length=EMBEDDING_DIM,
        use_jsonb=True,
    )


def reset_collection(doc_id: str, strategy: str) -> None:
    store = get_vectorstore(doc_id, strategy)
    store.delete_collection()
    store.create_collection()
    with get_engine().begin() as conn:
        conn.execute(
            text("delete from parent_chunks where doc_id = :d and strategy = :s"),
            {"d": doc_id, "s": strategy},
        )
        conn.execute(
            text("delete from ingest_runs where doc_id = :d and strategy = :s"),
            {"d": doc_id, "s": strategy},
        )


def ensure_hnsw_index() -> None:
    """langchain_postgres keeps every collection in one table, so one HNSW index covers them all."""
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "create index if not exists langchain_pg_embedding_hnsw on langchain_pg_embedding "
                "using hnsw (embedding vector_cosine_ops)"
            )
        )


# ---------- parent chunks ----------

def save_parents(doc_id: str, strategy: str, parents: list[dict]) -> None:
    if not parents:
        return
    rows = [{**p, "doc_id": doc_id, "strategy": strategy} for p in parents]
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "insert into parent_chunks (id, doc_id, strategy, section_path, page_start, page_end, text) "
                "values (:id, :doc_id, :strategy, :section_path, :page_start, :page_end, :text) "
                "on conflict (id) do update set text = excluded.text, "
                "section_path = excluded.section_path, page_start = excluded.page_start, "
                "page_end = excluded.page_end"
            ),
            rows,
        )


def get_parents(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                "select id, section_path, page_start, page_end, text "
                "from parent_chunks where id = any(:ids)"
            ),
            {"ids": list(ids)},
        )
        return {r.id: dict(r._mapping) for r in result}


# ---------- ingest runs ----------

def record_run(doc_id: str, strategy: str, chunk_count: int, page_count: int, parser_version: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "insert into ingest_runs (doc_id, strategy, chunk_count, page_count, parser_version) "
                "values (:d, :s, :c, :p, :v)"
            ),
            {"d": doc_id, "s": strategy, "c": chunk_count, "p": page_count, "v": parser_version},
        )


def latest_runs() -> dict[tuple[str, str], dict]:
    """Latest ingest run per (doc_id, strategy)."""
    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                "select distinct on (doc_id, strategy) doc_id, strategy, chunk_count, page_count, created_at "
                "from ingest_runs order by doc_id, strategy, created_at desc"
            )
        )
        return {(r.doc_id, r.strategy): dict(r._mapping) for r in result}


def load_all_chunks(doc_id: str, strategy: str) -> tuple[list[Document], np.ndarray]:
    """Every chunk and its embedding in a collection. Loaded once at startup so queries
    run in memory (BM25 + cosine) instead of paying Supabase round trips per request."""
    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                "select e.id, e.document, e.cmetadata, e.embedding::text as embedding "
                "from langchain_pg_embedding e "
                "join langchain_pg_collection c on c.uuid = e.collection_id "
                "where c.name = :n order by e.id"
            ),
            {"n": collection_name(doc_id, strategy)},
        ).all()
    docs = [Document(id=r.id, page_content=r.document, metadata=r.cmetadata or {}) for r in result]
    vectors = np.array([json.loads(r.embedding) for r in result], dtype=np.float32).reshape(len(result), -1)
    return docs, vectors


def load_parents(doc_id: str, strategy: str) -> dict[str, dict]:
    with get_engine().connect() as conn:
        result = conn.execute(
            text(
                "select id, section_path, page_start, page_end, text "
                "from parent_chunks where doc_id = :d and strategy = :s"
            ),
            {"d": doc_id, "s": strategy},
        )
        return {r.id: dict(r._mapping) for r in result}
