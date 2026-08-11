"""Tests for the response-shaping helpers.

Separate from test_api.py because these are pure functions over what the agent wrote.
They need no database, so they must not skip when there is no ingested corpus.
"""

from __future__ import annotations

from apps.api.schemas import resolve_citations, rfc_url


class TestRfcUrl:
    def test_links_to_the_document_without_a_section(self) -> None:
        assert rfc_url(9110) == "https://www.rfc-editor.org/rfc/rfc9110.html"

    def test_deep_links_to_a_section(self) -> None:
        assert rfc_url(9110, "7.2") == "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2"


class TestResolveCitations:
    def test_a_section_citation_becomes_a_deep_link(self) -> None:
        (resolved,) = resolve_citations(["RFC 9110 Section 7.2"])
        assert resolved.rfc_number == 9110
        assert resolved.section_number == "7.2"
        assert resolved.source_url.endswith("rfc9110.html#section-7.2")

    def test_a_bare_citation_links_to_the_document(self) -> None:
        (resolved,) = resolve_citations(["RFC 9110"])
        assert resolved.section_number is None
        assert resolved.source_url.endswith("rfc9110.html")

    def test_a_bare_citation_yields_to_a_precise_one(self) -> None:
        """Both name RFC 9110, and only one of them tells the reader where to look."""
        resolved = resolve_citations(["RFC 9110", "RFC 9110 Section 7.2"])
        assert [c.citation for c in resolved] == ["RFC 9110 Section 7.2"]

    def test_a_bare_citation_survives_when_nothing_is_more_precise(self) -> None:
        resolved = resolve_citations(["RFC 9110 Section 7.2", "RFC 2616"])
        assert [c.citation for c in resolved] == ["RFC 9110 Section 7.2", "RFC 2616"]

    def test_a_deep_subsection_is_kept_whole(self) -> None:
        (resolved,) = resolve_citations(["RFC 9110 Section 15.5.18"])
        assert resolved.section_number == "15.5.18"

    def test_unparseable_text_is_dropped_rather_than_shown_as_a_dead_link(self) -> None:
        assert resolve_citations(["see the IANA registry"]) == []

    def test_order_is_the_order_the_answer_used(self) -> None:
        resolved = resolve_citations(["RFC 9112 Section 3.2", "RFC 9110 Section 7.2"])
        assert [c.rfc_number for c in resolved] == [9112, 9110]
