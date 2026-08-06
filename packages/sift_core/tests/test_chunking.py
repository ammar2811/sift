"""Tests for section-aware chunking.

The invariant worth most here is that a chunk never spans two sections: it is what
lets a citation name an exact section, and it is silent when it breaks.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from packages.sift_core.chunking import ChunkConfig, chunk_document, chunk_section
from packages.sift_core.rfc_index import RfcMeta
from packages.sift_core.rfc_parser import ParsedRfc, Section

META = RfcMeta(
    number=9110,
    title="HTTP Semantics",
    year=2022,
    status="INTERNET STANDARD",
    stream="IETF",
    page_count=194,
    abstract=None,
    obsoletes=(2616,),
)


def _section(number: str | None, title: str, text: str) -> Section:
    return Section(number=number, title=title, text=text, line_start=0)


def test_chunks_never_span_sections() -> None:
    doc = ParsedRfc(
        number=9110,
        sections=[
            _section("7.2", "Host and :authority", "Alpha body.\n\nBeta body."),
            _section("7.3", "Routing", "Gamma body.\n\nDelta body."),
        ],
        body="",
    )
    chunks = chunk_document(doc, META, ChunkConfig(target_chars=10_000, overlap_chars=0))
    for chunk in chunks:
        assert not ("Alpha" in chunk.text and "Gamma" in chunk.text)
    assert {c.section_number for c in chunks} == {"7.2", "7.3"}


def test_short_section_survives_as_its_own_chunk() -> None:
    """RFC 9110 Section 15.5.18 is two lines long and still needs to be citable."""
    section = _section("15.5.18", "417 Expectation Failed", "The 417 status code means...")
    chunks = chunk_section(section, META, ChunkConfig(min_chars=500), start_ordinal=0)
    assert len(chunks) == 1
    assert chunks[0].citation == "RFC 9110 Section 15.5.18"


def test_heading_context_is_embedded_but_not_stored_in_text() -> None:
    section = _section("7.2", "Host and :authority", "The Host header field...")
    chunk = chunk_section(section, META, ChunkConfig(), start_ordinal=0)[0]
    assert "Section 7.2" in chunk.embedding_text
    assert "HTTP Semantics" in chunk.embedding_text
    assert "Section 7.2" not in chunk.text


def test_heading_context_can_be_disabled() -> None:
    section = _section("7.2", "Host", "Body text here.")
    cfg = ChunkConfig(include_heading_context=False)
    chunk = chunk_section(section, META, cfg, start_ordinal=0)[0]
    assert chunk.embedding_text == chunk.text


def test_dedent_preserves_relative_indentation() -> None:
    """ABNF and ASCII diagrams lose meaning if interior indentation is flattened."""
    body = "   Host = uri-host\n     port = DIGIT\n   End of grammar."
    chunk = chunk_section(_section("4", "Grammar", body), META, ChunkConfig(), start_ordinal=0)[0]
    assert "Host = uri-host" in chunk.text
    assert "  port = DIGIT" in chunk.text  # two spaces of relative indent survive


def test_overlap_carries_tail_of_previous_chunk() -> None:
    paras = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(12))
    cfg = ChunkConfig(target_chars=120, overlap_chars=40, min_chars=10)
    chunks = chunk_section(_section("1", "Intro", paras), META, cfg, start_ordinal=0)
    assert len(chunks) > 2
    for prev, cur in itertools.pairwise(chunks):
        tail = prev.text[-cfg.overlap_chars :].strip()
        assert any(word in cur.text for word in tail.split()[:3])


def test_zero_overlap_produces_disjoint_chunks() -> None:
    paras = "\n\n".join(f"Para {i} text." for i in range(10))
    cfg = ChunkConfig(target_chars=60, overlap_chars=0, min_chars=5)
    chunks = chunk_section(_section("1", "Intro", paras), META, cfg, start_ordinal=0)
    joined = "".join(c.text for c in chunks)
    assert joined.count("Para 0 text.") == 1


def test_oversized_paragraph_is_split_on_line_boundaries() -> None:
    para = "\n".join(f"line {i} of a very long unbroken paragraph block" for i in range(80))
    cfg = ChunkConfig(target_chars=300, overlap_chars=0, max_chars=400, min_chars=10)
    chunks = chunk_section(_section("2", "Table", para), META, cfg, start_ordinal=0)
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.split("\n"):
            assert line == "" or line.startswith("line ")


def test_heading_only_parent_section_yields_no_chunks() -> None:
    assert chunk_section(_section("7", "Message Routing", ""), META, ChunkConfig(), 0) == []


def test_ordinals_are_contiguous_across_document() -> None:
    doc = ParsedRfc(
        number=9110,
        sections=[
            _section("1", "Intro", "Body one here.\n\nMore body one."),
            _section("2", "Next", "Body two here.\n\nMore body two."),
        ],
        body="",
    )
    chunks = chunk_document(doc, META, ChunkConfig(target_chars=40, overlap_chars=0, min_chars=5))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_fallback_chunks_documents_without_headings() -> None:
    """RFCs 822, 855 and 907 indent every line and expose no column-0 headings."""
    doc = ParsedRfc(number=822, sections=[], body="   Indented body with no headings at all.")
    chunks = chunk_document(doc, META, ChunkConfig(min_chars=5))
    assert len(chunks) == 1
    assert chunks[0].section_number is None


def test_content_hash_tracks_embedded_text() -> None:
    a = chunk_section(_section("1", "A", "Same body."), META, ChunkConfig(), 0)[0]
    b = chunk_section(_section("1", "A", "Same body."), META, ChunkConfig(), 0)[0]
    c = chunk_section(_section("1", "A", "Different body."), META, ChunkConfig(), 0)[0]
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash


def test_normative_flag_propagates_to_chunks() -> None:
    section = _section("2.2", "Requirements", "A client MUST NOT send this header.")
    chunk = chunk_section(section, META, ChunkConfig(), 0)[0]
    assert chunk.has_normative


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"overlap_chars": 800}, "overlap_chars"),
        ({"min_chars": 5000}, "min_chars"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkConfig(**kwargs)


def test_fingerprint_distinguishes_configs() -> None:
    assert ChunkConfig().fingerprint != ChunkConfig(target_chars=400).fingerprint
    assert ChunkConfig().fingerprint == ChunkConfig().fingerprint
