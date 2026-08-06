"""Turn parsed RFCs into embeddable chunks.

Two rules shape everything here.

**Chunks never cross a section boundary.** A chunk that spans the end of Section 7.2
and the start of 7.3 cannot be cited precisely, and precise citation ("RFC 9110
Section 7.2") is the product's most visible quality signal. Section boundaries are
therefore hard splits, and long sections are divided internally.

**Splits prefer paragraph boundaries.** RFC bodies are hard-wrapped at ~72 columns,
so splitting on character count alone routinely cuts mid-sentence. Paragraphs are
packed greedily up to the target size, and only a single paragraph larger than the
target is split by line.

``ChunkConfig`` holds every knob the optimization sweep varies. Ingest-time and
query-time both import from this module so the two can never disagree.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from packages.sift_core.rfc_index import RfcMeta
from packages.sift_core.rfc_parser import ParsedRfc, Section

_BLANK_LINE = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """Chunking parameters. Every field here is a variable in the sweep."""

    target_chars: int = 800
    overlap_chars: int = 80
    min_chars: int = 120
    max_chars: int = 2000
    include_heading_context: bool = True
    dedent: bool = True

    def __post_init__(self) -> None:
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars")
        if self.min_chars > self.target_chars:
            raise ValueError("min_chars must not exceed target_chars")

    @property
    def fingerprint(self) -> str:
        """Stable id for this configuration, recorded alongside eval results."""
        parts = (
            f"t{self.target_chars}",
            f"o{self.overlap_chars}",
            f"m{self.min_chars}",
            f"x{self.max_chars}",
            f"h{int(self.include_heading_context)}",
            f"d{int(self.dedent)}",
        )
        return "-".join(parts)


@dataclass(slots=True)
class Chunk:
    rfc_number: int
    ordinal: int
    text: str
    section_number: str | None
    section_title: str | None
    has_normative: bool
    char_start: int
    char_end: int
    heading_context: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.section_number:
            return f"RFC {self.rfc_number} Section {self.section_number}"
        if self.section_title:
            return f"RFC {self.rfc_number} ({self.section_title})"
        return f"RFC {self.rfc_number}"

    @property
    def content_hash(self) -> bytes:
        """Identity of the chunk's embedded content, for idempotent re-ingest."""
        return hashlib.blake2b(self.embedding_text.encode("utf-8"), digest_size=16).digest()

    @property
    def embedding_text(self) -> str:
        """What actually gets embedded - body text, optionally prefixed by its heading.

        Prefixing the heading gives an isolated chunk enough context to be retrievable
        on its own; a chunk reading only "It MUST NOT be sent" is close to useless
        without knowing which header field "it" is.
        """
        if self.heading_context:
            return f"{self.heading_context}\n\n{self.text}"
        return self.text


def _dedent(text: str) -> str:
    """Strip the uniform left margin while preserving relative indentation.

    RFC bodies are indented three spaces, but they also contain ASCII diagrams and
    ABNF where interior indentation is meaningful, so only the common prefix goes.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text
    margin = min(len(ln) - len(ln.lstrip(" ")) for ln in lines)
    if margin == 0:
        return text
    return "\n".join(ln[margin:] if ln.strip() else "" for ln in text.split("\n"))


def _paragraphs(text: str) -> list[str]:
    return [p.strip("\n") for p in _BLANK_LINE.split(text) if p.strip()]


def _split_long_paragraph(para: str, limit: int) -> Iterator[str]:
    """Split an oversized paragraph on line boundaries, never mid-line.

    Lines are the smallest unit that keeps ASCII tables and ABNF readable.
    """
    buf: list[str] = []
    size = 0
    for line in para.split("\n"):
        if buf and size + len(line) + 1 > limit:
            yield "\n".join(buf)
            buf, size = [], 0
        buf.append(line)
        size += len(line) + 1
    if buf:
        yield "\n".join(buf)


def _pack(units: list[str], cfg: ChunkConfig) -> list[str]:
    """Greedily pack paragraphs up to the target size, then apply overlap."""
    packed: list[str] = []
    buf: list[str] = []
    size = 0
    for unit in units:
        pieces = (
            list(_split_long_paragraph(unit, cfg.target_chars))
            if len(unit) > cfg.max_chars
            else [unit]
        )
        for piece in pieces:
            if buf and size + len(piece) + 2 > cfg.target_chars:
                packed.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(piece)
            size += len(piece) + 2
    if buf:
        packed.append("\n\n".join(buf))

    if cfg.overlap_chars <= 0 or len(packed) < 2:
        return packed

    # Overlap carries the tail of each chunk into the next, so a fact split across a
    # boundary survives in at least one chunk intact.
    with_overlap = [packed[0]]
    for prev, cur in itertools.pairwise(packed):
        tail = prev[-cfg.overlap_chars :]
        # Start the carried text at a word boundary to avoid a severed token.
        if (space := tail.find(" ")) > 0:
            tail = tail[space + 1 :]
        with_overlap.append(f"{tail.strip()}\n\n{cur}" if tail.strip() else cur)
    return with_overlap


def chunk_section(
    section: Section, meta: RfcMeta, cfg: ChunkConfig, start_ordinal: int
) -> list[Chunk]:
    """Split one section into chunks. Returns [] for heading-only parent sections."""
    body = _dedent(section.text) if cfg.dedent else section.text
    if not body.strip():
        return []

    heading = None
    if cfg.include_heading_context:
        heading = f"RFC {meta.number}: {meta.title}\n{section.label}"

    texts = _pack(_paragraphs(body), cfg)

    chunks: list[Chunk] = []
    cursor = section.char_start
    for i, text in enumerate(texts):
        # Drop trailing scraps, but never drop a section's only chunk - a short
        # section such as "15.5.18. 417 Expectation Failed" is still worth citing.
        if len(text) < cfg.min_chars and len(texts) > 1 and i == len(texts) - 1:
            if chunks:
                chunks[-1].text = f"{chunks[-1].text}\n\n{text}"
                chunks[-1].char_end = cursor + len(text)
            continue
        chunks.append(
            Chunk(
                rfc_number=meta.number,
                ordinal=start_ordinal + len(chunks),
                text=text,
                section_number=section.number,
                section_title=section.title,
                has_normative=section.has_normative,
                char_start=cursor,
                char_end=cursor + len(text),
                heading_context=heading,
                metadata={
                    "status": meta.status,
                    "year": meta.year,
                    "area": meta.area,
                    "wg": meta.wg,
                    "is_current": meta.is_current,
                    "obsoleted_by": list(meta.obsoleted_by),
                    "updates": list(meta.updates),
                    "depth": section.depth,
                },
            )
        )
        cursor += len(text)
    return chunks


def chunk_document(doc: ParsedRfc, meta: RfcMeta, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Chunk a whole RFC, section by section."""
    cfg = cfg or ChunkConfig()
    chunks: list[Chunk] = []
    for section in doc.sections:
        chunks.extend(chunk_section(section, meta, cfg, start_ordinal=len(chunks)))

    # A handful of very early RFCs (822, 855, 907) indent every line and so expose no
    # column-0 headings at all. Rather than lose them, chunk the body as one section.
    if not chunks and doc.body.strip():
        fallback = Section(number=None, title=meta.title, text=doc.body, line_start=0, char_start=0)
        chunks = chunk_section(fallback, meta, cfg, start_ordinal=0)
    return chunks
