"""Tests for golden-set parsing and label matching.

The subsection-matching rule carries real weight: labelling ``9110:7.2`` and retrieving
``9110:7.2.1`` is a correct retrieval, and scoring it as a miss would understate recall
across the whole evaluation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.golden_set import (
    GOLDEN_DIR,
    GoldenSet,
    Question,
    QuestionType,
    SectionRef,
    load_golden_set,
)


def test_section_ref_parses_rfc_and_section() -> None:
    ref = SectionRef.parse("9110:7.2")
    assert ref.rfc_number == 9110
    assert ref.section == "7.2"


def test_section_ref_parses_whole_document() -> None:
    ref = SectionRef.parse("768")
    assert ref.rfc_number == 768
    assert ref.section is None


def test_whole_document_ref_matches_any_section() -> None:
    ref = SectionRef.parse("768")
    assert ref.matches(768, "1.2")
    assert ref.matches(768, None)
    assert not ref.matches(769, "1.2")


def test_subsection_counts_as_a_match() -> None:
    """Retrieving 7.2.1 when 7.2 was labelled is correct, not a miss."""
    ref = SectionRef.parse("9110:7.2")
    assert ref.matches(9110, "7.2")
    assert ref.matches(9110, "7.2.1")
    assert ref.matches(9110, "7.2.1.3")


def test_sibling_section_is_not_a_match() -> None:
    """7.2 must not match 7.20 - prefix matching has to respect the dot."""
    ref = SectionRef.parse("9110:7.2")
    assert not ref.matches(9110, "7.20")
    assert not ref.matches(9110, "7.3")
    assert not ref.matches(9110, "7")


def test_wrong_rfc_never_matches() -> None:
    assert not SectionRef.parse("9110:7.2").matches(2616, "7.2")


def test_missing_section_does_not_match_a_labelled_section() -> None:
    assert not SectionRef.parse("9110:7.2").matches(9110, None)


def test_section_ref_roundtrips_to_string() -> None:
    assert str(SectionRef.parse("9110:7.2")) == "9110:7.2"
    assert str(SectionRef.parse("768")) == "768"


def test_unanswerable_questions_are_not_scoreable() -> None:
    question = Question(
        id="u1", question="?", type=QuestionType.UNANSWERABLE, reference_answer="no"
    )
    assert not question.is_scoreable


def test_answerable_question_with_labels_is_scoreable() -> None:
    question = Question(
        id="a1",
        question="?",
        type=QuestionType.FACTUAL,
        reference_answer="yes",
        relevant=(SectionRef(9110, "7.2"),),
    )
    assert question.is_scoreable
    assert question.is_relevant(9110, "7.2.1")


def _write(tmp_path: Path, body: str) -> Path:
    (tmp_path / "q.yaml").write_text(body)
    return tmp_path


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
questions:
  - id: dup
    question: a
    type: factual
    reference_answer: a
    relevant: ["9110:7.2"]
  - id: dup
    question: b
    type: factual
    reference_answer: b
    relevant: ["9110:7.3"]
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_set(directory)


def test_answerable_question_without_labels_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
questions:
  - id: nolabels
    question: a
    type: factual
    reference_answer: a
""",
    )
    with pytest.raises(ValueError, match="need relevant sections"):
        load_golden_set(directory)


def test_unanswerable_question_with_labels_is_rejected(tmp_path: Path) -> None:
    directory = _write(
        tmp_path,
        """
questions:
  - id: bad
    question: a
    type: unanswerable
    reference_answer: a
    relevant: ["9110:7.2"]
""",
    )
    with pytest.raises(ValueError, match="cannot have labels"):
        load_golden_set(directory)


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return load_golden_set(GOLDEN_DIR)


class TestShippedGoldenSet:
    """Guards on the real question set, so its shape cannot silently drift."""

    def test_has_sixty_questions(self, golden: GoldenSet) -> None:
        assert len(golden) == 60

    def test_covers_every_question_type(self, golden: GoldenSet) -> None:
        for kind in QuestionType:
            assert golden.by_type(kind), f"no {kind.value} questions"

    def test_includes_a_meaningful_share_of_unanswerable(self, golden: GoldenSet) -> None:
        """Abstention is a headline claim; it needs enough questions to mean something."""
        assert len(golden.by_type(QuestionType.UNANSWERABLE)) >= 8

    def test_cross_document_questions_point_at_current_specs(self, golden: GoldenSet) -> None:
        """Labels must name the successor, not the obsolete document being asked about."""
        obsolete = {2616, 7230, 7231, 7234, 5246, 793}
        for question in golden.by_type(QuestionType.CROSS_DOCUMENT):
            labelled = {ref.rfc_number for ref in question.relevant}
            assert not labelled & obsolete, f"{question.id} labels an obsolete RFC"

    def test_every_answerable_question_has_a_reference_answer(self, golden: GoldenSet) -> None:
        for question in golden:
            assert question.reference_answer, f"{question.id} has no reference answer"

    def test_ids_are_unique_and_prefixed_by_type(self, golden: GoldenSet) -> None:
        prefixes = {
            QuestionType.FACTUAL: "fact-",
            QuestionType.NORMATIVE: "norm-",
            QuestionType.CROSS_DOCUMENT: "xdoc-",
            QuestionType.UNANSWERABLE: "unans-",
        }
        for question in golden:
            assert question.id.startswith(prefixes[question.type]), question.id
