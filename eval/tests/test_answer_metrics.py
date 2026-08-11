"""Tests for the answer metrics.

These are the numbers a README table will quote, so they are tested against hand-worked
cases rather than trusted to read correctly.
"""

from __future__ import annotations

import pytest

from eval.answer_metrics import (
    citation_precision,
    citation_recall,
    cited_refs,
    is_grounded,
    mention_coverage,
    uncited_rate,
)
from eval.golden_set import SectionRef

HOST = SectionRef(9110, "7.2")
STATUS_417 = SectionRef(9110, "15.5.18")


class TestCitedRefs:
    def test_parses_a_section_citation(self) -> None:
        assert cited_refs(["RFC 9110 Section 7.2"]) == [SectionRef(9110, "7.2")]

    def test_parses_a_bare_citation(self) -> None:
        assert cited_refs(["RFC 9110"]) == [SectionRef(9110, None)]

    def test_drops_text_that_is_not_a_citation(self) -> None:
        assert cited_refs(["see the registry"]) == []


class TestGrounding:
    def test_citing_a_labelled_section_is_grounded(self) -> None:
        assert is_grounded(["RFC 9110 Section 7.2"], [HOST])

    def test_citing_a_subsection_of_a_labelled_section_is_grounded(self) -> None:
        """The answer lives there too, so a more specific citation is not a miss."""
        assert is_grounded(["RFC 9110 Section 7.2.1"], [HOST])

    def test_citing_the_wrong_rfc_is_not_grounded(self) -> None:
        assert not is_grounded(["RFC 2616 Section 14.23"], [HOST])

    def test_citing_the_wrong_section_is_not_grounded(self) -> None:
        assert not is_grounded(["RFC 9110 Section 15.5.18"], [HOST])

    def test_citing_nothing_is_not_grounded(self) -> None:
        assert not is_grounded([], [HOST])

    def test_one_good_citation_among_several_grounds_the_answer(self) -> None:
        assert is_grounded(["RFC 2616", "RFC 9110 Section 7.2"], [HOST])


class TestCitationPrecision:
    def test_all_citations_labelled(self) -> None:
        assert citation_precision(["RFC 9110 Section 7.2"], [HOST]) == 1.0

    def test_half_the_citations_labelled(self) -> None:
        cited = ["RFC 9110 Section 7.2", "RFC 2616 Section 14.23"]
        assert citation_precision(cited, [HOST]) == pytest.approx(0.5)

    def test_no_citations_scores_zero_rather_than_dividing_by_zero(self) -> None:
        assert citation_precision([], [HOST]) == 0.0


class TestCitationRecall:
    def test_citing_every_labelled_section(self) -> None:
        cited = ["RFC 9110 Section 7.2", "RFC 9110 Section 15.5.18"]
        assert citation_recall(cited, [HOST, STATUS_417]) == 1.0

    def test_citing_one_of_two(self) -> None:
        assert citation_recall(["RFC 9110 Section 7.2"], [HOST, STATUS_417]) == pytest.approx(0.5)

    def test_extra_citations_do_not_raise_recall(self) -> None:
        cited = ["RFC 9110 Section 7.2", "RFC 2616", "RFC 9112 Section 3.2"]
        assert citation_recall(cited, [HOST, STATUS_417]) == pytest.approx(0.5)


class TestMentionCoverage:
    def test_all_terms_present(self) -> None:
        assert mention_coverage("the host and port information", ("host", "port")) == 1.0

    def test_matching_ignores_case(self) -> None:
        assert mention_coverage("Expectation Failed", ("expectation failed",)) == 1.0

    def test_partial_coverage(self) -> None:
        assert mention_coverage("the host only", ("host", "port")) == pytest.approx(0.5)

    def test_a_question_with_no_terms_is_not_penalised(self) -> None:
        assert mention_coverage("anything", ()) == 1.0


class TestUncitedRate:
    def test_an_assertion_without_a_citation_counts(self) -> None:
        assert uncited_rate([(False, False)]) == 1.0

    def test_an_abstention_without_a_citation_does_not_count(self) -> None:
        """Declining is correct behaviour, not the failure this measures."""
        assert uncited_rate([(False, True)]) == 0.0

    def test_a_cited_assertion_does_not_count(self) -> None:
        assert uncited_rate([(True, False)]) == 0.0

    def test_mixed(self) -> None:
        assert uncited_rate([(False, False), (True, False), (False, True), (True, False)]) == 0.25

    def test_empty_is_zero_rather_than_an_error(self) -> None:
        assert uncited_rate([]) == 0.0
