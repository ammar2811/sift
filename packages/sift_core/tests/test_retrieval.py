"""Integration tests for hybrid retrieval against a real Postgres."""

from __future__ import annotations

import random
from typing import Any

import psycopg
import pytest

from packages.sift_core.retrieval import (
    RRF_K,
    SearchFilters,
    SearchMode,
    get_section,
    search,
)

from .conftest import DIM, requires_corpus, requires_db

pytestmark = [requires_db, requires_corpus]


@pytest.fixture
def vector() -> list[float]:
    random.seed(3)
    return [random.random() for _ in range(DIM)]


def test_keyword_search_finds_the_right_section(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    conn, version_id = seeded_db
    hits = search(conn, version_id, query="417 Expectation Failed", mode="keyword", k=5)
    assert any(h.rfc_number == 9110 and h.section_number == "15.5.18" for h in hits)


def test_section_titles_are_weighted_above_body_text(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    """ "Host" appears in many bodies; the section titled Host should still win."""
    conn, version_id = seeded_db
    hits = search(conn, version_id, query="Host header field", mode="keyword", k=5)
    titles = [(h.section_title or "").lower() for h in hits[:3]]
    assert any("host" in t for t in titles)


def test_hybrid_fuses_both_retrievers(
    seeded_db: tuple[psycopg.Connection[Any], int], vector: list[float]
) -> None:
    conn, version_id = seeded_db
    hits = search(
        conn, version_id, query="Host header field", embedding=vector, mode="hybrid", k=10
    )
    assert any(h.dense_rank is not None for h in hits)
    assert any(h.keyword_rank is not None for h in hits)


def test_rrf_score_matches_the_formula(
    seeded_db: tuple[psycopg.Connection[Any], int], vector: list[float]
) -> None:
    conn, version_id = seeded_db
    hits = search(
        conn, version_id, query="Host header field", embedding=vector, mode="hybrid", k=10
    )
    for hit in hits:
        expected = 0.0
        if hit.dense_rank is not None:
            expected += 1 / (RRF_K + hit.dense_rank)
        if hit.keyword_rank is not None:
            expected += 1 / (RRF_K + hit.keyword_rank)
        assert hit.score == pytest.approx(expected, rel=1e-6)


def test_results_are_ordered_by_score(
    seeded_db: tuple[psycopg.Connection[Any], int], vector: list[float]
) -> None:
    conn, version_id = seeded_db
    hits = search(conn, version_id, query="header", embedding=vector, mode="hybrid", k=10)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_current_only_filter_excludes_obsoleted_documents(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    """RFC 2616 was obsoleted; RFC 9110 was not."""
    conn, version_id = seeded_db
    unfiltered = search(conn, version_id, query="Host header field", mode="keyword", k=20)
    assert any(h.rfc_number == 2616 for h in unfiltered)

    filtered = search(
        conn,
        version_id,
        query="Host header field",
        mode="keyword",
        k=20,
        filters=SearchFilters(current_only=True),
    )
    assert filtered
    assert all(h.rfc_number != 2616 for h in filtered)


def test_normative_filter_restricts_to_requirement_bearing_chunks(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    conn, version_id = seeded_db
    hits = search(
        conn,
        version_id,
        query="header",
        mode="keyword",
        k=10,
        filters=SearchFilters(normative_only=True),
    )
    assert hits
    assert all(h.metadata is not None for h in hits)


def test_rfc_number_filter_scopes_to_one_document(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    conn, version_id = seeded_db
    hits = search(
        conn,
        version_id,
        query="header",
        mode="keyword",
        k=10,
        filters=SearchFilters(rfc_numbers=(9110,)),
    )
    assert hits
    assert {h.rfc_number for h in hits} == {9110}


def test_get_section_returns_chunks_in_document_order(
    seeded_db: tuple[psycopg.Connection[Any], int],
) -> None:
    conn, version_id = seeded_db
    hits = get_section(conn, version_id, 9110, "7.2")
    assert hits
    assert all(h.section_number == "7.2" for h in hits)
    assert "Host" in hits[0].text


def test_citation_and_source_label(seeded_db: tuple[psycopg.Connection[Any], int]) -> None:
    conn, version_id = seeded_db
    hit = get_section(conn, version_id, 9110, "7.2")[0]
    assert hit.citation == "RFC 9110 Section 7.2"
    assert "HTTP Semantics" in hit.source_label


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [(SearchMode.DENSE, {"query": "x"}), (SearchMode.KEYWORD, {"embedding": [0.0] * DIM})],
)
def test_missing_inputs_are_rejected(
    seeded_db: tuple[psycopg.Connection[Any], int], mode: SearchMode, kwargs: dict[str, Any]
) -> None:
    conn, version_id = seeded_db
    with pytest.raises(ValueError, match="requires"):
        search(conn, version_id, mode=mode, k=1, **kwargs)


def test_k_limits_result_count(
    seeded_db: tuple[psycopg.Connection[Any], int], vector: list[float]
) -> None:
    conn, version_id = seeded_db
    assert len(search(conn, version_id, query="the", embedding=vector, mode="hybrid", k=3)) <= 3
