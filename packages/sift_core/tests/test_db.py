"""Tests for migration and vector-dimension handling.

The dimension guard exists because ``CREATE TABLE IF NOT EXISTS`` cannot retype a
column: without the guard, changing embedding dimension leaves a table that rejects
every insert, and the failure appears as a wall of per-document errors far from its
cause.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import psycopg
import pytest

from packages.sift_core import db

from .conftest import requires_db

pytestmark = requires_db


@pytest.fixture
def schema() -> Iterator[psycopg.Connection[Any]]:
    """A private schema so these tests never touch the development corpus."""
    conn = psycopg.connect(db.dsn(), row_factory=psycopg.rows.dict_row)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS sift_dbtest CASCADE")
        cur.execute("CREATE SCHEMA sift_dbtest")
        cur.execute("SET search_path TO sift_dbtest, public")
    conn.commit()
    yield conn
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS sift_dbtest CASCADE")
    conn.commit()
    conn.close()


def test_migrate_is_idempotent(schema: psycopg.Connection[Any]) -> None:
    db.migrate(schema, dimensions=32)
    db.migrate(schema, dimensions=32)
    assert db.existing_vector_type(schema) == "halfvec(32)"


def test_existing_vector_type_is_scoped_to_the_current_schema(
    schema: psycopg.Connection[Any],
) -> None:
    """The lookup must not resolve through search_path into another schema.

    `public` normally holds a chunks table of a different dimension; resolving to it
    would report a conflict that does not exist for a fresh schema.
    """
    assert db.existing_vector_type(schema) is None
    db.migrate(schema, dimensions=32)
    assert db.existing_vector_type(schema) == "halfvec(32)"


def test_dimension_change_is_refused_by_default(schema: psycopg.Connection[Any]) -> None:
    db.migrate(schema, dimensions=32)
    with pytest.raises(db.VectorDimensionMismatch, match="halfvec"):
        db.migrate(schema, dimensions=64)


def test_dimension_change_is_allowed_with_reset(schema: psycopg.Connection[Any]) -> None:
    db.migrate(schema, dimensions=32)
    db.migrate(schema, dimensions=64, recreate_vectors=True)
    assert db.existing_vector_type(schema) == "halfvec(64)"


def test_reset_clears_chunks_and_embedded_flags(schema: psycopg.Connection[Any]) -> None:
    """Vectors of one dimension are meaningless under another, so they must go."""
    from packages.sift_core.chunking import Chunk, ChunkConfig
    from packages.sift_core.rfc_index import RfcMeta

    db.migrate(schema, dimensions=4)
    meta = RfcMeta(
        number=9110,
        title="HTTP Semantics",
        year=2022,
        status="INTERNET STANDARD",
        stream="IETF",
        page_count=1,
        abstract=None,
    )
    db.upsert_documents(schema, [meta])
    version_id = db.ensure_corpus_version(schema, ChunkConfig(), dimensions=4)
    chunk = Chunk(
        rfc_number=9110,
        ordinal=0,
        text="body",
        section_number="7.2",
        section_title="Host",
        has_normative=False,
        char_start=0,
        char_end=4,
    )
    db.replace_document_chunks(schema, version_id, 9110, [chunk], [[0.1, 0.2, 0.3, 0.4]])
    assert db.corpus_stats(schema, version_id)["chunks"] == 1

    db.migrate(schema, dimensions=8, recreate_vectors=True)
    assert db.corpus_stats(schema, version_id)["chunks"] == 0
    with schema.cursor() as cur:
        cur.execute("SELECT is_embedded FROM documents WHERE number = 9110")
        row = cur.fetchone()
    assert row is not None and row["is_embedded"] is False


def test_only_one_corpus_version_can_be_active(schema: psycopg.Connection[Any]) -> None:
    from packages.sift_core.chunking import ChunkConfig

    db.migrate(schema, dimensions=8)
    first = db.ensure_corpus_version(schema, ChunkConfig(target_chars=400), dimensions=8)
    second = db.ensure_corpus_version(schema, ChunkConfig(target_chars=800), dimensions=8)

    db.activate_corpus_version(schema, first)
    assert db.active_version_id(schema) == first
    db.activate_corpus_version(schema, second)
    assert db.active_version_id(schema) == second

    with schema.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM corpus_versions WHERE is_active")
        row = cur.fetchone()
    assert row is not None and row["n"] == 1


def test_replace_document_chunks_rejects_mismatched_lengths(
    schema: psycopg.Connection[Any],
) -> None:
    from packages.sift_core.chunking import Chunk, ChunkConfig
    from packages.sift_core.rfc_index import RfcMeta

    db.migrate(schema, dimensions=4)
    db.upsert_documents(
        schema,
        [RfcMeta(9110, "HTTP", 2022, "INTERNET STANDARD", "IETF", 1, None)],
    )
    version_id = db.ensure_corpus_version(schema, ChunkConfig(), dimensions=4)
    chunk = Chunk(
        rfc_number=9110,
        ordinal=0,
        text="body",
        section_number=None,
        section_title=None,
        has_normative=False,
        char_start=0,
        char_end=4,
    )
    with pytest.raises(ValueError, match="embeddings"):
        db.replace_document_chunks(schema, version_id, 9110, [chunk], [])


def test_ingested_rfc_numbers_supports_resume(schema: psycopg.Connection[Any]) -> None:
    from packages.sift_core.chunking import Chunk, ChunkConfig
    from packages.sift_core.rfc_index import RfcMeta

    db.migrate(schema, dimensions=4)
    db.upsert_documents(
        schema,
        [
            RfcMeta(9110, "HTTP", 2022, "INTERNET STANDARD", "IETF", 1, None),
            RfcMeta(2616, "HTTP/1.1", 1999, "DRAFT STANDARD", "IETF", 1, None),
        ],
    )
    version_id = db.ensure_corpus_version(schema, ChunkConfig(), dimensions=4)
    assert db.ingested_rfc_numbers(schema, version_id) == set()

    chunk = Chunk(
        rfc_number=9110,
        ordinal=0,
        text="body",
        section_number=None,
        section_title=None,
        has_normative=False,
        char_start=0,
        char_end=4,
    )
    db.replace_document_chunks(schema, version_id, 9110, [chunk], [[0.1, 0.2, 0.3, 0.4]])
    assert db.ingested_rfc_numbers(schema, version_id) == {9110}
