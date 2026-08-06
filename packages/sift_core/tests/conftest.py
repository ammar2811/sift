"""Fixtures for tests that need a real Postgres with pgvector.

These are integration tests, not unit tests: RRF fusion, the `tsvector` weighting and
the halfvec distance operator are all database behaviour, and mocking them would test
the mock. They skip cleanly when no database is reachable so the unit suite still
runs anywhere.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from packages.sift_core import db
from packages.sift_core.chunking import ChunkConfig, chunk_document
from packages.sift_core.rfc_index import RfcMeta, load_index
from packages.sift_core.rfc_parser import parse_rfc

DIM = 64  # small vectors keep the fixture fast; fusion logic is dimension-agnostic
CORPUS = Path(__file__).resolve().parents[3] / "data"
FIXTURE_RFCS = (9110, 2616, 9293)


def _has_database() -> bool:
    try:
        with psycopg.connect(db.dsn(), connect_timeout=2):
            return True
    except psycopg.Error:
        return False


requires_db = pytest.mark.skipif(not _has_database(), reason="no Postgres reachable")
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "rfc-index.xml").exists(),
    reason="corpus not downloaded (python -m apps.worker.fetch_corpus)",
)


@pytest.fixture(scope="session")
def seeded_db() -> Iterator[tuple[psycopg.Connection[Any], int]]:
    """A throwaway schema holding a few real RFCs with deterministic fake vectors."""
    random.seed(11)
    conn = psycopg.connect(db.dsn(), row_factory=psycopg.rows.dict_row)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS sift_test CASCADE")
        cur.execute("CREATE SCHEMA sift_test")
        cur.execute("SET search_path TO sift_test, public")
    conn.commit()

    db.migrate(conn, dimensions=DIM)
    metas = load_index(CORPUS / "rfc-index.xml")
    by_num: dict[int, RfcMeta] = {m.number: m for m in metas}
    db.upsert_documents(conn, [by_num[n] for n in FIXTURE_RFCS])

    cfg = ChunkConfig()
    version_id = db.ensure_corpus_version(conn, cfg, dimensions=DIM)
    db.activate_corpus_version(conn, version_id)

    for number in FIXTURE_RFCS:
        path = CORPUS / "rfc-cache" / f"rfc{number}.txt"
        if not path.exists():
            continue
        doc = parse_rfc(number, path.read_text(errors="replace"))
        chunks = chunk_document(doc, by_num[number], cfg)
        vectors = [[random.random() for _ in range(DIM)] for _ in chunks]
        db.replace_document_chunks(conn, version_id, number, chunks, vectors)

    yield conn, version_id

    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS sift_test CASCADE")
    conn.commit()
    conn.close()
