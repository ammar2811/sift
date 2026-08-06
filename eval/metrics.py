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
    """Discounted cumulative gain against the best ordering of the same results.

    The ideal is the retrieved relevance sorted descending - every relevant item moved
    to the front - which bounds the result at 1.0 by construction.

    It deliberately is *not* ``[True] * total_relevant``. ``total_relevant`` counts
    labelled sections, but one label is satisfied by several retrieved chunks, since a
    label of ``9110:7.2`` is matched by 7.2, 7.2.1 and 7.2.1.3 alike. Three relevant
    chunks against a single label produced a DCG of 2.13 over an "ideal" of 1.0, and
    an nDCG above 1.0, which is not a meaningful quantity.

    Whether the whole labelled set was found is what recall measures; this measures
    how well what was found is ordered.
    """
    if k <= 0 or total_relevant <= 0:
        return 0.0
    window = list(relevance[:k])
    ideal = dcg(sorted(window, reverse=True), k)
    return dcg(window, k) / ideal if ideal > 0 else 0.0


def hit_rate(relevance: Sequence[bool], k: int) -> float:
    """1.0 when anything relevant appears in the top ``k``."""
    return 1.0 if any(relevance[:k]) else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
