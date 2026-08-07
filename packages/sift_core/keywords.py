"""Selecting which words in a question are worth searching for.

PostgreSQL's ``ts_rank_cd`` has no inverse-document-frequency term. A match on a word
occurring in 0.009% of the corpus counts exactly as much as a match on "response". At
7,364 chunks that rarely decided anything; at 129,109 it decided almost everything, and
the correct answer to "what must a client do when it receives a 417 response?" was
outranked 9.20 to 1.40 by a DHCP section that merely repeats "client", "must",
"receive" and "response".

Since the ranking function will not weight by rarity, the query is filtered by rarity
instead: words too common to distinguish anything are dropped before ranking, leaving
the ones that actually identify the answer. This is the part BM25 would otherwise
supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A word in more than this share of the corpus cannot narrow anything down. 5% is
# generous - "response" appears in far more than that - and is deliberately not so
# aggressive that ordinary questions lose every term.
DEFAULT_MAX_DOC_RATIO = 0.05

# Never let a question collapse to nothing, and never let it stay so broad that the
# rare words are drowned again.
MIN_TERMS = 1
MAX_TERMS = 8


@dataclass(frozen=True, slots=True)
class ScoredTerm:
    lexeme: str
    ndoc: int
    idf: float

    @property
    def is_discriminative(self) -> bool:
        return self.ndoc > 0


def inverse_document_frequency(ndoc: int, total_docs: int) -> float:
    """Standard smoothed IDF.

    A word absent from the corpus scores highest: it cannot match anything, but its
    presence in the question is a strong signal about intent, and keeping it costs
    nothing because it simply matches no rows.
    """
    if total_docs <= 0:
        return 0.0
    return math.log(1.0 + (total_docs / max(ndoc, 1)))


def score_terms(doc_counts: dict[str, int], total_docs: int) -> list[ScoredTerm]:
    scored = [
        ScoredTerm(lexeme, ndoc, inverse_document_frequency(ndoc, total_docs))
        for lexeme, ndoc in doc_counts.items()
    ]
    return sorted(scored, key=lambda t: t.idf, reverse=True)


def select_discriminative(
    doc_counts: dict[str, int],
    total_docs: int,
    *,
    max_doc_ratio: float = DEFAULT_MAX_DOC_RATIO,
    max_terms: int = MAX_TERMS,
) -> list[str]:
    """Keep the words worth searching for, most distinctive first.

    Words above ``max_doc_ratio`` of the corpus are dropped. If that would leave
    nothing - a question made entirely of common words - the most distinctive terms
    are kept anyway, because an over-broad query still beats an empty one.
    """
    if not doc_counts:
        return []

    scored = score_terms(doc_counts, total_docs)
    ceiling = max_doc_ratio * total_docs if total_docs > 0 else float("inf")

    kept = [t.lexeme for t in scored if t.ndoc <= ceiling][:max_terms]
    if len(kept) >= MIN_TERMS:
        return kept

    # Everything was common. Fall back to the least common of them.
    return [t.lexeme for t in scored[:max_terms]]
