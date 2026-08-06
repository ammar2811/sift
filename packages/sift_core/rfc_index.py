"""Parsing and selection over the RFC Editor's ``rfc-index.xml``.

The index is the source of truth for RFC-level metadata and, importantly, for the
supersession graph (``obsoletes`` / ``obsoleted-by`` / ``updates`` / ``updated-by``).

A deliberate asymmetry runs through this module and the rest of Sift: metadata for
*every* RFC is cheap to store, but embedding a full text is not. So we ingest the
complete graph (~9,800 entries) while embedding only a selected subset. That way
``resolve_current_spec`` can always walk a supersession chain to the document in
force today, even when the endpoints of that chain were never embedded.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

NS = "{https://www.rfc-editor.org/rfc-index}"

INDEX_URL = "https://www.rfc-editor.org/rfc-index.xml"
TEXT_URL = "https://www.rfc-editor.org/rfc/rfc{number}.txt"

STANDARDS_TRACK = frozenset({"PROPOSED STANDARD", "DRAFT STANDARD", "INTERNET STANDARD"})
MATURE_STANDARDS = frozenset({"DRAFT STANDARD", "INTERNET STANDARD"})
BCP = frozenset({"BEST CURRENT PRACTICE"})

_DOC_ID = re.compile(r"^RFC(\d+)$")


def _doc_number(doc_id: str) -> int | None:
    """``"RFC9110"`` -> ``9110``. Non-RFC ids (STD/BCP/FYI aliases) yield ``None``."""
    m = _DOC_ID.match(doc_id.strip())
    return int(m.group(1)) if m else None


@dataclass(frozen=True, slots=True)
class RfcMeta:
    number: int
    title: str
    year: int
    status: str
    stream: str
    page_count: int
    abstract: str | None
    authors: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    area: str | None = None
    wg: str | None = None
    obsoletes: tuple[int, ...] = ()
    obsoleted_by: tuple[int, ...] = ()
    updates: tuple[int, ...] = ()
    updated_by: tuple[int, ...] = ()
    has_text: bool = True

    @property
    def doc_id(self) -> str:
        return f"RFC{self.number}"

    @property
    def is_current(self) -> bool:
        """True when nothing has obsoleted this RFC - i.e. it is still in force."""
        return not self.obsoleted_by


def _refs(entry: ET.Element, tag: str) -> tuple[int, ...]:
    node = entry.find(f"{NS}{tag}")
    if node is None:
        return ()
    numbers = (_doc_number(d.text or "") for d in node.findall(f"{NS}doc-id"))
    return tuple(n for n in numbers if n is not None)


def _text_of(entry: ET.Element, *path: str) -> str | None:
    node: ET.Element | None = entry
    for part in path:
        if node is None:
            return None
        node = node.find(f"{NS}{part}")
    return node.text.strip() if node is not None and node.text else None


def _parse_entry(entry: ET.Element) -> RfcMeta | None:
    number = _doc_number(entry.findtext(f"{NS}doc-id") or "")
    if number is None:
        return None

    year_text = _text_of(entry, "date", "year")
    pages_text = entry.findtext(f"{NS}page-count")
    fmt = entry.find(f"{NS}format")
    formats = {f.text for f in fmt.findall(f"{NS}file-format")} if fmt is not None else set()

    # The abstract is wrapped in <p> elements; join their text.
    abstract_node = entry.find(f"{NS}abstract")
    abstract = None
    if abstract_node is not None:
        paras = [(p.text or "").strip() for p in abstract_node]
        abstract = "\n\n".join(p for p in paras if p) or None

    kw_node = entry.find(f"{NS}keywords")
    keywords = (
        tuple((k.text or "").strip() for k in kw_node if (k.text or "").strip())
        if kw_node is not None
        else ()
    )

    return RfcMeta(
        number=number,
        title=(entry.findtext(f"{NS}title") or "").strip(),
        year=int(year_text) if year_text and year_text.isdigit() else 0,
        status=(entry.findtext(f"{NS}current-status") or "UNKNOWN").strip(),
        stream=(entry.findtext(f"{NS}stream") or "").strip(),
        page_count=int(pages_text) if pages_text and pages_text.isdigit() else 0,
        abstract=abstract,
        authors=tuple(
            name.strip()
            for a in entry.findall(f"{NS}author")
            if (name := a.findtext(f"{NS}name") or "")
        ),
        keywords=keywords,
        area=entry.findtext(f"{NS}area"),
        wg=entry.findtext(f"{NS}wg_acronym"),
        obsoletes=_refs(entry, "obsoletes"),
        obsoleted_by=_refs(entry, "obsoleted-by"),
        updates=_refs(entry, "updates"),
        updated_by=_refs(entry, "updated-by"),
        has_text="TXT" in formats,
    )


def load_index(path: Path) -> list[RfcMeta]:
    """Parse every ``rfc-entry`` in the index. Skips the 188 never-issued numbers."""
    root = ET.parse(path).getroot()
    parsed = (_parse_entry(e) for e in root.findall(f"{NS}rfc-entry"))
    return sorted((m for m in parsed if m is not None), key=lambda m: m.number)


@dataclass(frozen=True, slots=True)
class CorpusSelector:
    """Which RFCs get their full text embedded.

    Defaults select every RFC that is *currently in force* as a mature standard or
    best current practice, plus recent Proposed Standards that nothing has obsoleted.
    ``proposed_since`` is the tuning knob: it trades corpus breadth against embedding
    cost and re-ingestion time, which is the binding constraint during the
    optimization sweep.
    """

    mature: frozenset[str] = MATURE_STANDARDS | BCP
    proposed_since: int | None = 2020
    require_text: bool = True
    include: frozenset[int] = field(default_factory=frozenset)
    exclude: frozenset[int] = field(default_factory=frozenset)

    def selects(self, m: RfcMeta) -> bool:
        if m.number in self.exclude:
            return False
        if m.number in self.include:
            return True
        if self.require_text and not m.has_text:
            return False
        if m.status in self.mature:
            return True
        return (
            self.proposed_since is not None
            and m.status == "PROPOSED STANDARD"
            and m.is_current
            and m.year >= self.proposed_since
        )

    def apply(self, metas: Iterable[RfcMeta]) -> list[RfcMeta]:
        return [m for m in metas if self.selects(m)]


def resolve_current_spec(metas: dict[int, RfcMeta], number: int, max_hops: int = 16) -> int:
    """Follow ``obsoleted-by`` to the RFC in force today.

    Where an RFC was split into several successors (RFC 2616 -> 7230..7235), the
    lowest-numbered successor is followed; callers wanting the full fan-out should
    read ``obsoleted_by`` directly. Cycles and runaway chains stop at ``max_hops``.
    """
    seen: set[int] = set()
    current = number
    for _ in range(max_hops):
        meta = metas.get(current)
        if meta is None or not meta.obsoleted_by or current in seen:
            return current
        seen.add(current)
        current = min(meta.obsoleted_by)
    return current
