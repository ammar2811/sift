"""Metrics over a generated answer, computed without a model.

These are the half of answer quality that does not need a judge, and they are kept
apart from the judge for that reason: they are deterministic, free, and they cannot
drift when a model deployment changes underneath them.

A caveat that belongs on every number here rather than in a footnote. The golden set's
``relevant`` field lists sections sufficient to answer the question, not every section
it would be *acceptable* to cite. An answer that cites a section the labels do not name
is not thereby wrong - RFC prose is repetitive and several sections often support the
same claim. So citation precision as computed here is a lower bound on the real thing,
and it is reported as one. Citation recall and grounding do not have this problem:
citing a labelled section is unambiguously correct.
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.golden_set import SectionRef
from packages.sift_core.agent import parse_citation


def cited_refs(citations: Sequence[str]) -> list[SectionRef]:
    """The citations an answer carried, as comparable references."""
    refs: list[SectionRef] = []
    for text in citations:
        parsed = parse_citation(text)
        if parsed is not None:
            refs.append(SectionRef(parsed[0], parsed[1]))
    return refs


def _matches(cited: SectionRef, labelled: Sequence[SectionRef]) -> bool:
    return any(ref.matches(cited.rfc_number, cited.section) for ref in labelled)


def is_grounded(citations: Sequence[str], labelled: Sequence[SectionRef]) -> bool:
    """Whether the answer cited at least one section the labels name.

    The weakest useful claim, and the one least sensitive to the labels being
    non-exhaustive: an answer that cites nothing labelled may still be right, but
    nothing here can tell that it is.
    """
    return any(_matches(ref, labelled) for ref in cited_refs(citations))


def citation_precision(citations: Sequence[str], labelled: Sequence[SectionRef]) -> float:
    """Fraction of cited sections that the labels name. A lower bound - see module docs."""
    refs = cited_refs(citations)
    if not refs:
        return 0.0
    return sum(1 for ref in refs if _matches(ref, labelled)) / len(refs)


def citation_recall(citations: Sequence[str], labelled: Sequence[SectionRef]) -> float:
    """Fraction of labelled sections the answer cited."""
    if not labelled:
        return 0.0
    refs = cited_refs(citations)
    found = sum(
        1 for label in labelled if any(label.matches(r.rfc_number, r.section) for r in refs)
    )
    return found / len(labelled)


def mention_coverage(answer: str, must_mention: Sequence[str]) -> float:
    """Fraction of the question's key terms the answer actually contains.

    ``must_mention`` guards the labels rather than the answer - it asserts the terms
    appear in the cited *sources*. Reused here it is a cheap, if blunt, check that the
    answer engaged with the substance instead of gesturing at the right document.
    """
    if not must_mention:
        return 1.0
    lowered = answer.lower()
    return sum(1 for term in must_mention if term.lower() in lowered) / len(must_mention)


def uncited_rate(results: Sequence[tuple[bool, bool]]) -> float:
    """Share of answers that asserted something while citing nothing.

    Each element is ``(has_citation, refused)``. An abstention with no citation is
    correct behaviour and is not counted; an assertion with no citation is the exact
    failure the agent's redo path exists to prevent, so it is measured separately from
    every quality question.
    """
    if not results:
        return 0.0
    return sum(1 for cited, refused in results if not cited and not refused) / len(results)
