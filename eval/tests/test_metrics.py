"""Tests for retrieval metrics, with expected values computed by hand.

Every headline number in the README comes from these functions, so each case states
the arithmetic rather than asserting against whatever the implementation happens to
return.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    dcg,
    hit_rate,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_against_all_labelled_sections() -> None:
    # Two of three relevant sections found in the top 5 -> 2/3.
    relevance = [True, False, True, False, False]
    assert recall_at_k(relevance, total_relevant=3, k=5) == pytest.approx(2 / 3)


def test_recall_is_not_inflated_by_partial_retrieval() -> None:
    """Finding one of three relevant sections must not score 1.0."""
    assert recall_at_k([True], total_relevant=3, k=10) == pytest.approx(1 / 3)


def test_recall_respects_the_cutoff() -> None:
    relevance = [False, False, False, True]
    assert recall_at_k(relevance, total_relevant=1, k=3) == 0.0
    assert recall_at_k(relevance, total_relevant=1, k=4) == 1.0


def test_recall_is_capped_at_one() -> None:
    assert recall_at_k([True, True, True], total_relevant=2, k=3) == 1.0


def test_recall_with_no_relevant_sections_is_zero() -> None:
    assert recall_at_k([True], total_relevant=0, k=5) == 0.0


def test_precision_at_k() -> None:
    assert precision_at_k([True, False, True, False], k=4) == 0.5
    assert precision_at_k([True, True], k=2) == 1.0
    assert precision_at_k([], k=0) == 0.0


def test_reciprocal_rank_uses_the_first_relevant_position() -> None:
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True, True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_dcg_applies_log2_discount() -> None:
    # Positions 1 and 3 -> 1/log2(2) + 1/log2(4) = 1.0 + 0.5
    assert dcg([True, False, True], k=3) == pytest.approx(1.0 + 0.5)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k([True, True, False], total_relevant=2, k=10) == pytest.approx(1.0)


def test_ndcg_penalises_a_late_relevant_result() -> None:
    perfect = ndcg_at_k([True, False, False], total_relevant=1, k=10)
    late = ndcg_at_k([False, False, True], total_relevant=1, k=10)
    assert perfect == pytest.approx(1.0)
    # 1/log2(4) divided by the ideal 1.0
    assert late == pytest.approx(1 / math.log2(4))
    assert late < perfect


def test_ndcg_ideal_is_capped_by_k() -> None:
    """With 5 relevant sections but k=2, retrieving 2 is a perfect score."""
    assert ndcg_at_k([True, True], total_relevant=5, k=2) == pytest.approx(1.0)


def test_ndcg_never_exceeds_one_when_a_label_matches_several_chunks() -> None:
    """Subsection matching means one label can be satisfied by several chunks.

    A label of "9110:7.2" is matched by 7.2, 7.2.1 and 7.2.1.3. Computing the ideal
    from the label count then gave DCG 2.13 over an ideal of 1.0 - an nDCG above 1.0,
    which is not a meaningful quantity.
    """
    relevance = [True, True, True] + [False] * 7
    assert ndcg_at_k(relevance, total_relevant=1, k=10) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "relevance",
    [
        [True] * 10,
        [True, False, True, True, False],
        [False, True],
        [True],
        [False] * 5,
    ],
)
def test_ndcg_is_always_within_the_unit_interval(relevance: list[bool]) -> None:
    for total in (1, 2, 5):
        value = ndcg_at_k(relevance, total_relevant=total, k=10)
        assert 0.0 <= value <= 1.0, f"{relevance} with total={total} gave {value}"


def test_ndcg_still_rewards_better_ordering() -> None:
    """Bounding the metric must not flatten it into a constant."""
    early = ndcg_at_k([True, True, False, False], total_relevant=2, k=4)
    late = ndcg_at_k([False, False, True, True], total_relevant=2, k=4)
    assert early == pytest.approx(1.0)
    assert late < early


def test_ndcg_with_nothing_relevant_is_zero() -> None:
    assert ndcg_at_k([False, False], total_relevant=2, k=5) == 0.0
    assert ndcg_at_k([True], total_relevant=0, k=5) == 0.0


def test_hit_rate_is_binary() -> None:
    assert hit_rate([False, True, False], k=3) == 1.0
    assert hit_rate([False, True], k=1) == 0.0


def test_mean_handles_empty_input() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == 1.5
