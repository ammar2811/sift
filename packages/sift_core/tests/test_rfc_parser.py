"""Regression tests for the RFC text parser.

Each case here corresponds to a layout that actually broke the parser while it was
being built against the real corpus, so they are guards against specific regressions
rather than illustrations of the happy path.
"""

from __future__ import annotations

import pytest

from packages.sift_core.rfc_parser import parse_rfc, strip_page_furniture

PAGINATED = """\
Network Working Group                                          J. Postel
Request for Comments:  792                                           ISI
Updates:  RFCs 777, 760

                   INTERNET CONTROL MESSAGE PROTOCOL

Introduction

   The Internet Protocol (IP) is used for host-to-host datagram service.




                                                                [Page 1]
\x0c

                                                          September 1981
RFC 792



Message Formats

   ICMP messages are sent using the basic IP header.
"""

MODERN = """\
﻿Internet Engineering Task Force (IETF)                            T. Ito
Request for Comments: 9919                               SECOM CO., LTD.
Category: Standards Track                                     July 2026

Table of Contents

   1.  Introduction
   2.  Conventions
     2.1.  Requirements Language

1.  Introduction

   This document profiles OCSP.

2.  Conventions

2.1.  Requirements Language

   The key words "MUST" and "SHOULD NOT" are to be interpreted as
   described in BCP 14.
"""


def test_strips_page_footer_and_running_header() -> None:
    cleaned = strip_page_furniture(PAGINATED)
    assert "[Page 1]" not in cleaned
    assert "\x0c" not in cleaned
    # The running header here is inverted - date above the RFC number - which a
    # single-line "RFC N  Title  Date" pattern cannot match.
    assert "September 1981" not in cleaned
    assert not any(line.strip() == "RFC 792" for line in cleaned.splitlines())
    assert "ICMP messages are sent" in cleaned


def test_unnumbered_headings_are_sections() -> None:
    """Early RFCs number nothing; without these they parse to zero sections."""
    doc = parse_rfc(792, PAGINATED)
    titles = [s.title for s in doc.sections]
    assert "Introduction" in titles
    assert "Message Formats" in titles


def test_masthead_is_not_a_heading() -> None:
    doc = parse_rfc(792, PAGINATED)
    titles = [s.title for s in doc.sections]
    assert not any("Network Working Group" in t for t in titles)
    assert not any("Request for Comments" in t for t in titles)
    assert not any("Updates" in t for t in titles)


def test_diagrams_and_definition_lists_are_not_headings() -> None:
    """Column-0 lines laid out in columns are artwork, not section titles.

    RFC 9605 indexed "Alice |  (per frame)  (per packet) |" as a section title, so a
    citation rendered as that string in the UI and the tsvector gave it the weight it
    reserves for real titles. 198 titles across 33 documents looked like this.
    """
    raw = """\
Network Working Group                                          T. Example
Request for Comments: 9999                                    Example Inc

Introduction

   Normal body text that follows a real heading.

Alice |          (per frame)         (per packet)   |        |       |
      |               ^                   ^         |        |       |

RDATA           a variable length string of octets that describes it

   More body text.
"""
    titles = [s.title for s in parse_rfc(9999, raw).sections]
    assert "Introduction" in titles
    assert not any("|" in t for t in titles)
    assert not any(t.startswith("RDATA") for t in titles)


def test_table_of_contents_is_not_mistaken_for_body() -> None:
    doc = parse_rfc(9919, MODERN)
    numbers = [s.number for s in doc.numbered_sections]
    # Each section must appear once - from the body, not also from the ToC.
    assert numbers.count("1") == 1
    assert numbers.count("2.1") == 1


def test_section_text_belongs_to_its_own_heading() -> None:
    doc = parse_rfc(9919, MODERN)
    s = next(x for x in doc.numbered_sections if x.number == "2.1")
    assert "BCP 14" in s.text
    assert "profiles OCSP" not in s.text


def test_normative_keyword_detection() -> None:
    doc = parse_rfc(9919, MODERN)
    intro = next(x for x in doc.numbered_sections if x.number == "1")
    reqs = next(x for x in doc.numbered_sections if x.number == "2.1")
    assert reqs.has_normative
    assert not intro.has_normative


def test_boilerplate_sections_are_dropped() -> None:
    text = MODERN.replace("1.  Introduction\n\n   This document", "Copyright Notice\n\n   Blah")
    doc = parse_rfc(9919, text)
    assert not any(s.title.lower() == "copyright notice" for s in doc.sections)


def test_citation_label_is_human_readable() -> None:
    doc = parse_rfc(9919, MODERN)
    s = next(x for x in doc.numbered_sections if x.number == "2.1")
    assert s.label == "Section 2.1. Requirements Language"
    assert s.depth == 2


@pytest.mark.parametrize("number", ["1", "2", "2.1"])
def test_expected_sections_present(number: str) -> None:
    doc = parse_rfc(9919, MODERN)
    assert any(s.number == number for s in doc.numbered_sections)


def test_deeply_nested_numbering_is_preserved() -> None:
    """NFSv4.1 (RFC 8881) nests to 2.6.3.1.1.4 - depth must not be truncated."""
    text = "1.  Top\n\n   Body.\n\n2.6.3.1.1.4  Deep\n\n   Deep body.\n"
    doc = parse_rfc(8881, text)
    deep = next(x for x in doc.numbered_sections if x.number == "2.6.3.1.1.4")
    assert deep.depth == 6
    assert "Deep body." in deep.text
