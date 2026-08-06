"""Parse RFC plain text into a section tree.

Two publishing eras share the cache. Older RFCs are paginated: each page ends with a
``Author  [Page N]`` footer, a form feed, and a running ``RFC NNNN  Title  Date``
header. RFCs produced from the v3 XML toolchain are unpaginated and carry none of
that furniture. Both eras, however, put section headings hard against column 0 while
indenting body text by three spaces - that single invariant is what makes reliable
section extraction possible, and it is why this corpus was chosen.

Sections matter because they turn a citation into "RFC 9110 Section 8.3" rather than
"chunk 417", which is the most visible quality signal in the product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A heading is a column-0 line: a dotted number, optional trailing dot, whitespace,
# then a title. Body text is indented, so anchoring at column 0 excludes prose, and
# the table of contents is indented in every era, so it is excluded too.
_HEADING = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?[ \t]+(?P<title>\S.*?)\s*$")
_APPENDIX = re.compile(r"^(?:Appendix[ \t]+)?(?P<num>[A-Z](?:\.\d+)*)\.[ \t]+(?P<title>\S.*?)\s*$")
_UNNUMBERED = re.compile(r"^(?P<title>[A-Z][A-Za-z][^.]{2,70})$")

_PAGE_FOOTER = re.compile(r"^.*\[Page[ \t]+\d+\][ \t]*$")

# Running headers are not one fixed shape. Most read "RFC 9110  Title  June 2022",
# but older RFCs split them across two lines and sometimes invert the order, putting
# the date above the RFC number (see RFC 792). Rather than enumerate every layout,
# recognise the *ingredients* of a header line and drop a short run of them.
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_HEADER_INGREDIENT = re.compile(
    rf"(^|\s)(RFC[ \t]+\d+|(?:{_MONTHS})[ \t]+\d{{4}}|\[Page[ \t]+\d+\])(\s|$)"
)
_MAX_HEADER_LINES = 3

# Dotted leaders ("1. Scope ....... 5") only ever appear in a table of contents.
_TOC_LEADER = re.compile(r"\.{3,}\s*\d+\s*$")

# "Request for Comments: 792", "Category: Standards Track", "Obsoletes: 5019" - the
# masthead block, which sits at column 0 exactly like a heading does.
_FRONT_MATTER_LABEL = re.compile(r"^[A-Z][A-Za-z' ]{2,40}:[ \t]")
_DATE_LINE = re.compile(rf"^(?:{_MONTHS})[ \t]+\d{{1,2}},[ \t]*\d{{4}}\s*$")

# RFC 2119 / BCP 14 requirement keywords, which are what make a section normative.
_NORMATIVE = re.compile(
    r"\b(MUST NOT|SHALL NOT|SHOULD NOT|NOT RECOMMENDED|MUST|SHALL|SHOULD"
    r"|REQUIRED|RECOMMENDED|MAY|OPTIONAL)\b"
)

# Front matter that is boilerplate in every RFC and pure noise in a retrieval index.
_BOILERPLATE_TITLES = frozenset(
    {
        "status of this memo",
        "copyright notice",
        "table of contents",
        "full copyright statement",
        "intellectual property",
        "intellectual property statement",
        "acknowledgment",
        "acknowledgments",
        "acknowledgements",
        "author's address",
        "authors' addresses",
        "author information",
        "contributors",
        "index",
    }
)


@dataclass(slots=True)
class Section:
    number: str | None
    title: str
    text: str
    line_start: int
    char_start: int = 0
    char_end: int = 0

    @property
    def depth(self) -> int:
        return self.number.count(".") + 1 if self.number else 0

    @property
    def has_normative(self) -> bool:
        return bool(_NORMATIVE.search(self.text))

    @property
    def label(self) -> str:
        """Human citation form, e.g. ``Section 3.1. Transports``."""
        return f"Section {self.number}. {self.title}" if self.number else self.title


@dataclass(slots=True)
class ParsedRfc:
    number: int
    sections: list[Section] = field(default_factory=list)
    body: str = ""

    @property
    def numbered_sections(self) -> list[Section]:
        return [s for s in self.sections if s.number]


def strip_page_furniture(text: str) -> str:
    """Remove page footers, form feeds and running headers from paginated RFCs.

    Unpaginated RFCs contain none of these patterns, so this is a no-op for them and
    a single code path serves both eras.
    """
    out: list[str] = []
    lines = text.replace("\r\n", "\n").replace("﻿", "").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "\f" in line:
            # The form feed may share a line with surrounding text; keep that text
            # unless it is itself part of the header.
            remainder = line.replace("\f", "").strip()
            if remainder and not _HEADER_INGREDIENT.search(remainder):
                out.append(remainder)
            i += 1
            # Skip blank padding, then consume the short run of header lines that
            # follows, whatever order its parts appear in.
            while i < n and not lines[i].strip():
                i += 1
            consumed = 0
            while i < n and consumed < _MAX_HEADER_LINES and lines[i].strip():
                if not _HEADER_INGREDIENT.search(lines[i]):
                    break
                i += 1
                consumed += 1
                while i < n and not lines[i].strip():
                    i += 1
            continue
        if _PAGE_FOOTER.match(line) and "[Page" in line:
            i += 1
            continue
        out.append(line)
        i += 1

    # Collapse the runs of blank lines left behind by removed furniture.
    collapsed: list[str] = []
    blanks = 0
    for line in out:
        if line.strip():
            blanks = 0
            collapsed.append(line.rstrip())
        else:
            blanks += 1
            if blanks <= 2:
                collapsed.append("")
    return "\n".join(collapsed)


def _find_body_start(lines: list[str]) -> int:
    """Skip the header block, boilerplate and table of contents.

    Returns the index of the first line belonging to real content. Falls back to 0
    when no table of contents is present, which is common in very early RFCs.
    """
    toc_end = 0
    for i, line in enumerate(lines[:400]):
        if line.strip().lower().startswith("table of contents"):
            # Walk to the end of the ToC: the first column-0 heading after it.
            for j in range(i + 1, min(len(lines), i + 600)):
                stripped = lines[j]
                if _TOC_LEADER.search(stripped):
                    continue
                if stripped and not stripped[0].isspace() and _HEADING.match(stripped):
                    return j
            toc_end = i
            break
    return toc_end


def _next_nonblank(lines: list[str], i: int) -> str | None:
    for line in lines[i + 1 :]:
        if line.strip():
            return line
    return None


def _followed_by_body(lines: list[str], i: int, lookahead: int = 2) -> bool:
    """True when an indented line appears soon after ``i``.

    This separates real headings from the front-matter block ("Network Working
    Group", "Request for Comments: 792", "Updates: RFCs 777, 760"), which also sits
    at column 0 in older RFCs. A heading introduces indented body text; front matter
    is followed by more front matter.
    """
    seen = 0
    for line in lines[i + 1 : i + 1 + lookahead * 4]:
        if not line.strip():
            continue
        if line[0].isspace():
            return True
        seen += 1
        if seen >= lookahead:
            return False
    return False


def _is_heading(
    line: str, lines: list[str] | None = None, index: int = 0
) -> tuple[str | None, str] | None:
    """Classify a column-0 line as a numbered, appendix, or unnumbered heading."""
    if not line or line[0].isspace() or _TOC_LEADER.search(line):
        return None
    if _FRONT_MATTER_LABEL.match(line) or _DATE_LINE.match(line):
        return None
    if m := _HEADING.match(line):
        title = m.group("title")
        # "1. 2. 3." style enumerations and lines ending in a comma are prose.
        if title.endswith((",", ";")):
            return None
        return m.group("num"), title
    if m := _APPENDIX.match(line):
        return m.group("num"), m.group("title")
    if m := _UNNUMBERED.match(line):
        # Early RFCs (RFC 792, 822, 1984) number nothing and rely entirely on
        # unnumbered column-0 headings. Without these they parse to zero sections.
        title = m.group("title").strip()
        if _HEADER_INGREDIENT.search(title) or len(title.split()) > 10:
            return None
        if title.endswith((",", ";", ":")):
            return None
        if lines is not None:
            if not _followed_by_body(lines, index):
                return None
            # The masthead's first line ("Network Working Group") carries no colon of
            # its own, but the line under it always does.
            nxt = _next_nonblank(lines, index)
            if nxt is not None and _FRONT_MATTER_LABEL.match(nxt):
                return None
        return None, title
    return None


def parse_rfc(number: int, raw: str, *, drop_boilerplate: bool = True) -> ParsedRfc:
    """Parse one RFC's plain text into an ordered list of sections."""
    body = strip_page_furniture(raw)
    lines = body.split("\n")
    start = _find_body_start(lines)

    heads: list[tuple[int, str | None, str]] = []
    for i in range(start, len(lines)):
        if (h := _is_heading(lines[i], lines, i)) is not None:
            heads.append((i, h[0], h[1]))

    sections: list[Section] = []
    for idx, (line_no, num, title) in enumerate(heads):
        end = heads[idx + 1][0] if idx + 1 < len(heads) else len(lines)
        chunk_lines = lines[line_no + 1 : end]
        text = "\n".join(chunk_lines).strip("\n")
        if drop_boilerplate and title.strip().lower() in _BOILERPLATE_TITLES:
            continue
        if not text.strip():
            # A heading with no body of its own is a parent (e.g. "3." above "3.1.");
            # keep it so the tree stays navigable, but it will not produce chunks.
            text = ""
        sections.append(Section(number=num, title=title.strip(), text=text, line_start=line_no))

    # Character offsets let a citation point back into the exact source span.
    cursor = 0
    for s in sections:
        s.char_start = cursor
        cursor += len(s.text)
        s.char_end = cursor

    return ParsedRfc(number=number, sections=sections, body=body)
