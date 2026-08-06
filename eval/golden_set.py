"""The labelled question set and its schema.

Each question names the RFC sections that *should* be retrieved, expressed as
``"9110:7.2"``. Retrieval is then scored against those labels without an LLM in the
loop, which means recall, MRR and nDCG can be measured, and the sweep can be run,
before any generation model exists.

Question types are tracked separately because they fail for different reasons and a
single aggregate number hides that:

``factual``
    A specific value stated in one place. Tests basic lookup.
``normative``
    A requirement expressed with RFC 2119 keywords. Tests whether MUST/SHOULD survive
    chunking and whether the right *requirement* is retrieved rather than adjacent
    prose.
``cross_document``
    Needs the supersession graph - the answer lives in a document that obsoleted the
    one the question names. Single-shot retrieval is expected to do poorly here, which
    is what justifies the agentic layer.
``unanswerable``
    The corpus cannot answer it. Scored on abstention, never on retrieval. Most
    portfolio RAG evaluations omit these entirely.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

GOLDEN_DIR = Path(__file__).parent / "golden"


class QuestionType(StrEnum):
    FACTUAL = "factual"
    NORMATIVE = "normative"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True, slots=True)
class SectionRef:
    """A citable location: RFC number plus section, e.g. ``9110:7.2``."""

    rfc_number: int
    section: str | None = None

    @classmethod
    def parse(cls, raw: str) -> SectionRef:
        text = str(raw).strip()
        if ":" in text:
            number, section = text.split(":", 1)
            return cls(int(number), section.strip() or None)
        return cls(int(text), None)

    def matches(self, rfc_number: int, section: str | None) -> bool:
        """A hit counts when it is in the right RFC and inside the right section.

        A labelled section of ``7.2`` is satisfied by ``7.2`` or by a subsection such
        as ``7.2.1``, since the answer genuinely lives there. Labelling only the exact
        section would penalise correct retrieval of a more specific passage.
        """
        if rfc_number != self.rfc_number:
            return False
        if self.section is None:
            return True
        if section is None:
            return False
        return section == self.section or section.startswith(f"{self.section}.")

    def __str__(self) -> str:
        return f"{self.rfc_number}:{self.section}" if self.section else str(self.rfc_number)


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    question: str
    type: QuestionType
    reference_answer: str
    relevant: tuple[SectionRef, ...] = ()
    # Terms that must appear in the labelled *source* sections, verified by
    # `python -m eval.validate_golden`. These guard the labels, not the answer: a
    # question whose key term is absent from every section it points at is mislabelled.
    # Note that an RFC almost never states its own number in body prose, so "which RFC"
    # expectations belong in `relevant`, not here.
    must_mention: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def is_scoreable(self) -> bool:
        """Unanswerable questions have no correct retrieval to measure."""
        return self.type is not QuestionType.UNANSWERABLE and bool(self.relevant)

    def is_relevant(self, rfc_number: int, section: str | None) -> bool:
        return any(ref.matches(rfc_number, section) for ref in self.relevant)


@dataclass(slots=True)
class GoldenSet:
    questions: list[Question] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    @property
    def scoreable(self) -> list[Question]:
        return [q for q in self.questions if q.is_scoreable]

    def by_type(self, kind: QuestionType) -> list[Question]:
        return [q for q in self.questions if q.type is kind]

    def rfc_numbers(self) -> set[int]:
        return {ref.rfc_number for q in self.questions for ref in q.relevant}


def _parse_question(raw: dict[str, Any]) -> Question:
    return Question(
        id=str(raw["id"]),
        question=str(raw["question"]).strip(),
        type=QuestionType(raw["type"]),
        reference_answer=str(raw.get("reference_answer", "")).strip(),
        relevant=tuple(SectionRef.parse(r) for r in raw.get("relevant", ())),
        must_mention=tuple(str(m) for m in raw.get("must_mention", ())),
        notes=raw.get("notes"),
    )


def load_golden_set(directory: Path | None = None) -> GoldenSet:
    """Load and validate every YAML file in the golden directory."""
    directory = directory or GOLDEN_DIR
    questions: list[Question] = []
    seen: set[str] = set()

    for path in sorted(directory.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        for raw in payload.get("questions", []):
            question = _parse_question(raw)
            if question.id in seen:
                raise ValueError(f"duplicate question id {question.id!r} in {path.name}")
            seen.add(question.id)
            if question.type is QuestionType.UNANSWERABLE and question.relevant:
                raise ValueError(f"{question.id}: unanswerable questions cannot have labels")
            if question.type is not QuestionType.UNANSWERABLE and not question.relevant:
                raise ValueError(f"{question.id}: answerable questions need relevant sections")
            questions.append(question)

    return GoldenSet(questions)
