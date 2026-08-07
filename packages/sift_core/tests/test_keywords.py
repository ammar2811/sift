"""Tests for discriminative term selection.

These encode the finding that motivated the module: at 129,109 chunks, "417" occurs in
12 of them and "must" in 27,109, yet PostgreSQL's ts_rank_cd weights a match on each
identically. Selecting on rarity before ranking is what restores the distinction.
"""

from __future__ import annotations

import pytest

from packages.sift_core.keywords import (
    inverse_document_frequency,
    score_terms,
    select_discriminative,
)

# Real frequencies measured on the full corpus.
CORPUS = 129_109
REAL_COUNTS = {
    "417": 12,
    "client": 14_041,
    "must": 27_109,
    "receiv": 15_467,
    "respons": 11_373,
}


def test_rarer_terms_score_higher() -> None:
    assert inverse_document_frequency(12, CORPUS) > inverse_document_frequency(27_109, CORPUS)


def test_absent_terms_are_not_penalised() -> None:
    """A word absent from the corpus simply matches nothing; it must not error."""
    assert inverse_document_frequency(0, CORPUS) > 0


def test_zero_corpus_is_safe() -> None:
    assert inverse_document_frequency(5, 0) == 0.0


def test_the_rare_token_survives_and_the_common_ones_do_not() -> None:
    """The exact case that was failing: only '417' should reach the query."""
    assert select_discriminative(REAL_COUNTS, CORPUS) == ["417"]


def test_terms_are_ordered_most_distinctive_first() -> None:
    ordered = [t.lexeme for t in score_terms(REAL_COUNTS, CORPUS)]
    assert ordered[0] == "417"
    assert ordered[-1] == "must"


def test_an_all_common_question_still_produces_a_query() -> None:
    """Dropping every term would return nothing at all, which is worse than a broad query."""
    counts = {"client": 14_041, "must": 27_109, "respons": 11_373}
    kept = select_discriminative(counts, CORPUS)
    assert kept
    assert kept[0] == "respons"  # the least common of them


def test_empty_input() -> None:
    assert select_discriminative({}, CORPUS) == []


def test_term_count_is_capped() -> None:
    counts = {f"term{i}": 1 for i in range(50)}
    assert len(select_discriminative(counts, CORPUS)) <= 8


@pytest.mark.parametrize("ratio", [0.001, 0.05, 0.5])
def test_ratio_controls_strictness(ratio: float) -> None:
    kept = select_discriminative(REAL_COUNTS, CORPUS, max_doc_ratio=ratio)
    assert "417" in kept
    if ratio >= 0.5:
        assert len(kept) > 1
