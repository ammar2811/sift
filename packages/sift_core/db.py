"""Database access: connection pooling, migrations, and vector index management.

Pooling matters more than usual here. The target deployment is a Postgres Flexible
Server B1ms with a single vCore and a low connection ceiling, so every API replica
shares one small pool rather than opening connections per request.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from packages.sift_core.chunking import Chunk, ChunkConfig
from packages.sift_core.rfc_index import RfcMeta

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# halfvec halves storage against float32 at negligible recall cost, which is what
# keeps a 129k-chunk index inside the 2 GiB of a B1ms.
DEFAULT_DIMENSIONS = 768
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

_pool: ConnectionPool | None = None


def dsn() -> str:
    return os.environ.get("SIFT_DATABASE_URL", "postgresql://sift:sift@localhost:5432/sift")


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            dsn(),
            min_size=1,
            max_size=int(os.environ.get("SIFT_DB_POOL_MAX", "8")),
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection[Any]]:
    with get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def vector_type(dimensions: int = DEFAULT_DIMENSIONS, *, half: bool = True) -> str:
    return f"halfvec({dimensions})" if half else f"vector({dimensions})"


class VectorDimensionMismatch(RuntimeError):
    """The chunks table's vector column does not match the requested dimension."""


def existing_vector_type(conn: psycopg.Connection[Any]) -> str | None:
    """The declared type of ``chunks.embedding`` in the *current* schema.

    Deliberately not ``to_regclass('chunks')``: that resolves through ``search_path``,
    so migrating a fresh schema would inspect some other schema's table and report a
    dimension conflict that does not exist. Returns None when this schema has no
    chunks table yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) AS type_name
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = 'chunks'
              AND n.nspname = current_schema()
              AND a.attname = 'embedding'
              AND a.attnum > 0
            """
        )
        row = cur.fetchone()
    return str(row["type_name"]) if row else None


def _recreate_embedding_column(conn: psycopg.Connection[Any], wanted: str) -> None:
    """Retype the vector column, discarding existing vectors.

    Vectors of a different dimension carry no meaning under the new one, so there is
    nothing to preserve - every chunk has to be re-embedded regardless.
    """
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        cur.execute("TRUNCATE TABLE chunks")
        cur.execute(
            sql.SQL("ALTER TABLE chunks ALTER COLUMN embedding TYPE {}").format(sql.SQL(wanted))
        )
        cur.execute("UPDATE documents SET is_embedded = false, ingested_at = NULL")
    conn.commit()


def migrate(
    conn: psycopg.Connection[Any],
    dimensions: int = DEFAULT_DIMENSIONS,
    *,
    half: bool = True,
    recreate_vectors: bool = False,
) -> None:
    """Apply the schema. Idempotent - every statement is IF NOT EXISTS.

    That idempotence has one sharp edge: ``CREATE TABLE IF NOT EXISTS`` cannot change
    an existing column's type, so switching embedding dimension would leave a table
    that silently rejects every insert. Since the optimization sweep varies dimension,
    the mismatch is detected here and reported rather than discovered as a wall of
    per-document failures.
    """
    wanted = vector_type(dimensions, half=half)
    existing = existing_vector_type(conn)
    if existing is not None and existing != wanted:
        if not recreate_vectors:
            raise VectorDimensionMismatch(
                f"chunks.embedding is {existing} but this run needs {wanted}. "
                "Vectors of one dimension are meaningless under another, so the "
                "column must be retyped and the corpus re-embedded: rerun with "
                "--reset-vectors (destroys all stored chunks)."
            )
        _recreate_embedding_column(conn, wanted)

    ddl = SCHEMA_PATH.read_text().replace("${VECTOR_TYPE}", wanted)
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def upsert_documents(conn: psycopg.Connection[Any], metas: Sequence[RfcMeta]) -> int:
    """Insert or refresh RFC metadata for the whole index."""
    rows = [
        (
            m.number,
            m.title,
            m.year,
            m.status,
            m.stream,
            m.page_count,
            m.abstract,
            list(m.authors),
            list(m.keywords),
            m.area,
            m.wg,
            list(m.obsoletes),
            list(m.obsoleted_by),
            list(m.updates),
            list(m.updated_by),
            m.has_text,
        )
        for m in metas
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (number, title, year, status, stream, page_count,
                abstract, authors, keywords, area, wg, obsoletes, obsoleted_by,
                updates, updated_by, has_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (number) DO UPDATE SET
                title = EXCLUDED.title, year = EXCLUDED.year, status = EXCLUDED.status,
                stream = EXCLUDED.stream, page_count = EXCLUDED.page_count,
                abstract = EXCLUDED.abstract, authors = EXCLUDED.authors,
                keywords = EXCLUDED.keywords, area = EXCLUDED.area, wg = EXCLUDED.wg,
                obsoletes = EXCLUDED.obsoletes, obsoleted_by = EXCLUDED.obsoleted_by,
                updates = EXCLUDED.updates, updated_by = EXCLUDED.updated_by,
                has_text = EXCLUDED.has_text
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def ensure_corpus_version(
    conn: psycopg.Connection[Any],
    cfg: ChunkConfig,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_DIMENSIONS,
    scope: str = "default",
    notes: str | None = None,
) -> int:
    """Get or create the corpus version row for this configuration.

    ``scope`` names the document set. It is part of the identity because a number is
    only comparable against the same documents.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO corpus_versions
                (fingerprint, embedding_model, dimensions, scope, chunk_config, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (fingerprint, embedding_model, dimensions, scope)
                DO UPDATE SET notes = COALESCE(EXCLUDED.notes, corpus_versions.notes)
            RETURNING id
            """,
            (
                cfg.fingerprint,
                embedding_model,
                dimensions,
                scope,
                json.dumps(asdict(cfg)),
                notes,
            ),
        )
        row = cur.fetchone()
    conn.commit()
    assert row is not None
    return int(row["id"])


def activate_corpus_version(conn: psycopg.Connection[Any], version_id: int) -> None:
    """Point queries at one configuration. The partial unique index enforces one."""
    with conn.cursor() as cur:
        cur.execute("UPDATE corpus_versions SET is_active = false WHERE is_active")
        cur.execute("UPDATE corpus_versions SET is_active = true WHERE id = %s", (version_id,))
    conn.commit()


def active_version_id(conn: psycopg.Connection[Any]) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM corpus_versions WHERE is_active LIMIT 1")
        row = cur.fetchone()
    return int(row["id"]) if row else None


def replace_document_chunks(
    conn: psycopg.Connection[Any],
    version_id: int,
    rfc_number: int,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> int:
    """Write one document's chunks, replacing any previous ones.

    Delete-then-insert inside a single transaction is what makes re-ingestion
    idempotent: a rerun after a partial failure leaves exactly one copy of the
    document, never duplicates. The sweep re-ingests constantly, so this matters.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunks WHERE version_id = %s AND rfc_number = %s",
            (version_id, rfc_number),
        )
        cur.executemany(
            """
            INSERT INTO chunks (version_id, rfc_number, ordinal, section_number,
                section_title, text, embedding, has_normative, char_start, char_end,
                content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    version_id,
                    c.rfc_number,
                    c.ordinal,
                    c.section_number,
                    c.section_title,
                    c.text,
                    str(list(vec)),
                    c.has_normative,
                    c.char_start,
                    c.char_end,
                    c.content_hash,
                    json.dumps(c.metadata),
                )
                for c, vec in zip(chunks, embeddings, strict=True)
            ],
        )
        cur.execute(
            """
            UPDATE documents SET is_embedded = true, ingested_at = now()
            WHERE number = %s
            """,
            (rfc_number,),
        )
    conn.commit()
    return len(chunks)


def create_vector_index(
    conn: psycopg.Connection[Any],
    *,
    half: bool = True,
    m: int = 16,
    ef_construction: int = 64,
) -> None:
    """Build the HNSW index. Call after bulk load, never before.

    Building during ingestion slows the load substantially and produces a worse graph
    than a single build over the finished table.
    """
    ops = "halfvec_cosine_ops" if half else "vector_cosine_ops"
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks "
                "USING hnsw (embedding {}) WITH (m = {}, ef_construction = {})"
            ).format(sql.SQL(ops), sql.Literal(m), sql.Literal(ef_construction))
        )
    conn.commit()


def corpus_version(conn: psycopg.Connection[Any], version_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM corpus_versions WHERE id = %s", (version_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def ingested_rfc_numbers(conn: psycopg.Connection[Any], version_id: int) -> set[int]:
    """RFCs that already have chunks under this corpus version.

    Ingestion is idempotent, but redoing finished documents wastes the most expensive
    step in the pipeline. This lets an interrupted run resume where it stopped.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT rfc_number FROM chunks WHERE version_id = %s", (version_id,))
        return {int(row["rfc_number"]) for row in cur.fetchall()}


def build_lexeme_stats(conn: psycopg.Connection[Any], version_id: int) -> int:
    """Count how many chunks contain each word, for this corpus version.

    This is the inverse-document-frequency data PostgreSQL's own ranking lacks. Built
    once after ingestion because ``ts_stat`` scans every row, which is far too slow to
    do per query.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM lexeme_stats WHERE version_id = %s", (version_id,))
        cur.execute(
            """
            INSERT INTO lexeme_stats (version_id, lexeme, ndoc)
            SELECT %s, word, ndoc
            FROM ts_stat(
                format('SELECT tsv FROM chunks WHERE version_id = %%s', %s::text)
            )
            """,
            (version_id, version_id),
        )
        rows = cur.rowcount
    conn.commit()
    return int(rows)


def has_lexeme_stats(conn: psycopg.Connection[Any], version_id: int) -> bool:
    """Whether frequency data exists for this version.

    Worth asking separately, because an empty table is indistinguishable from "every
    one of these words is rare" once the counts come back defaulted to zero.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM lexeme_stats WHERE version_id = %s LIMIT 1", (version_id,))
        return cur.fetchone() is not None


def lexeme_document_counts(
    conn: psycopg.Connection[Any], version_id: int, lexemes: Sequence[str]
) -> dict[str, int]:
    """Document frequency for each supplied lexeme. Missing words count as 0."""
    if not lexemes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT lexeme, ndoc FROM lexeme_stats WHERE version_id = %s AND lexeme = ANY(%s)",
            (version_id, list(lexemes)),
        )
        found = {str(r["lexeme"]): int(r["ndoc"]) for r in cur.fetchall()}
    return {lexeme: found.get(lexeme, 0) for lexeme in lexemes}


def corpus_chunk_count(conn: psycopg.Connection[Any], version_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM chunks WHERE version_id = %s", (version_id,))
        row = cur.fetchone()
    return int(row["n"]) if row else 0


def corpus_stats(conn: psycopg.Connection[Any], version_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS chunks,
                   count(DISTINCT rfc_number) AS documents,
                   count(*) FILTER (WHERE section_number IS NOT NULL) AS with_section,
                   count(*) FILTER (WHERE has_normative) AS normative,
                   avg(length(text))::int AS mean_chars
            FROM chunks WHERE version_id = %s
            """,
            (version_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}
