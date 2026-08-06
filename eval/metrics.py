"""Retrieval metrics.

Deliberately dependency-free and separately tested. These numbers are the project's
central claim, so they should be verifiable by reading them rather than trusted
because a library produced them.

All three treat relevance as binary, which suits the labels: a section either
contains the answer or it does not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """Fraction of the relevant items that appear in the top ``k``.

    ``total_relevant`` is the number of labelled sections, not the number retrieved -
    otherwise a run that retrieves one of three relevant sections would score 1.0.
    """
    if total_relevant <= 0:
        return 0.0
    found = sum(1 for hit in relevance[:k] if hit)
    return min(found / total_relevant, 1.0)


def precision_at_k(relevance: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    window = relevance[:k]
    return sum(1 for hit in window if hit) / max(len(window), 1)


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1 / rank of the first relevant result; 0 when none is relevant.

    Rewards putting a correct result first, which is what matters when only the top
    few chunks reach the generation prompt.
    """
    for index, hit in enumerate(relevance, start=1):
        if hit:
            return 1.0 / index
    return 0.0


def dcg(relevance: Sequence[bool], k: int) -> float:
    return sum(1.0 / math.log2(i + 1) for i, hit in enumerate(relevance[:k], start=1) if hit)


def ndcg_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    """Discounted cumulative gain against the best achievable ordering.

    The ideal ranking places every relevant item first, capped by both ``k`` and the
    number of relevant items that exist.
    """
    ideal_hits = min(total_relevant, k)
    if ideal_hits <= 0:
        return 0.0
    ideal = dcg([True] * ideal_hits, k)
    return dcg(relevance, k) / ideal if ideal > 0 else 0.0


def hit_rate(relevance: Sequence[bool], k: int) -> float:
    """1.0 when anything relevant appears in the top ``k``."""
    return 1.0 if any(relevance[:k]) else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
