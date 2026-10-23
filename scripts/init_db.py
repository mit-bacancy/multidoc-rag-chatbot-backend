"""Check the Supabase connection, enable pgvector, create parent_chunks / ingest_runs.

    python -m scripts.init_db
"""

from sqlalchemy import text

from app import store


def main() -> None:
    store.init_schema()
    with store.get_engine().connect() as conn:
        version = conn.execute(text("select extversion from pg_extension where extname = 'vector'")).scalar()
        tables = conn.execute(
            text("select table_name from information_schema.tables where table_name in ('parent_chunks','ingest_runs')")
        ).scalars().all()
    print(f"DB OK | pgvector {version} | tables: {sorted(tables)}")


if __name__ == "__main__":
    main()
