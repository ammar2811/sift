"""Check the golden set against the corpus.

A labelled section that does not exist, or that does not contain the answer, silently
corrupts every metric computed from it - recall would be measured against an
unreachable target. This runs over the parsed corpus rather than the database, so it
works before anything has been ingested.

    python -m eval.validate_golden
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from eval.golden_set import GOLDEN_DIR, QuestionType, load_golden_set
from packages.sift_core.rfc_index import load_index
from packages.sift_core.rfc_parser import parse_rfc

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "rfc-cache"
INDEX = DATA / "rfc-index.xml"


def _obsolete_labels(golden: object) -> list[str]:
    """Labels pointing at a document some other RFC has superseded.

    Corpus selection keeps only what is currently in force, so a label naming a
    superseded document can never be retrieved - it scores the right answer as a total
    miss. Five TLS questions were labelled against RFC 8446 after RFC 9846 obsoleted it,
    and retrieval was returning the correct 9846 sections while measuring zero recall.
    """
    if not INDEX.exists():
        return []
    by_number = {m.number: m for m in load_index(INDEX)}
    problems = []
    for question in golden:  # type: ignore[attr-defined]
        for ref in question.relevant:
            meta = by_number.get(ref.rfc_number)
            if meta is not None and not meta.is_current:
                successors = ", ".join(str(n) for n in sorted(meta.obsoleted_by)) or "unknown"
                problems.append(
                    f"{question.id}: RFC{ref.rfc_number} is obsoleted by {successors} "
                    f"and is not in the corpus; relabel to the specification in force"
                )
    return problems


def _stale_reference_answers(golden: object) -> list[str]:
    """Reference answers that name a superseded RFC without naming its successor.

    The sibling check above guards ``relevant``, and that is where the TLS relabelling
    was applied. It could not catch ``unans-tls14``: unanswerable questions carry no
    labels, so its reference answer went on asserting that TLS 1.3 lives in RFC 8446
    long after RFC 9846 obsoleted it. The system then answered correctly, cited 9846,
    and was graded wrong by a judge comparing it against a stale reference.

    Naming an obsolete RFC is legitimate when the answer is *about* supersession - the
    cross-document references say "RFC 9846 ... obsoleting ... RFC 8446" and should. So
    the rule is not "never mention an obsolete RFC" but "if you do, name what replaced
    it", which is exactly the difference between describing history and being out of date.
    """
    if not INDEX.exists():
        return []
    by_number = {m.number: m for m in load_index(INDEX)}

    def successors(number: int) -> set[int]:
        """Every document downstream of this one, not only the direct replacements.

        Supersession runs in chains here - RFC 2616 was replaced by 7230-7235, which
        RFC 9110 then replaced - and a reference answer that jumps straight to 9110 has
        named the right specification. Checking only direct successors would flag it.
        """
        seen: set[int] = set()
        frontier = [number]
        while frontier:
            meta = by_number.get(frontier.pop())
            for nxt in meta.obsoleted_by if meta else ():
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    problems = []
    for question in golden:  # type: ignore[attr-defined]
        # "RFCs 7230 through 7235" is how a reference answer names a split successor set.
        cited = {int(n) for n in re.findall(r"RFCs?\s*(\d+)", question.reference_answer)}
        for number in sorted(cited):
            meta = by_number.get(number)
            if meta is None or meta.is_current:
                continue
            if cited & successors(number):
                continue
            replaced_by = ", ".join(str(n) for n in sorted(meta.obsoleted_by)) or "unknown"
            problems.append(
                f"{question.id}: reference_answer cites RFC{number}, obsoleted by "
                f"{replaced_by}, without naming any successor; update it or say what "
                f"replaced it"
            )
    return problems


def main() -> int:
    golden = load_golden_set(GOLDEN_DIR)
    problems: list[str] = _obsolete_labels(golden) + _stale_reference_answers(golden)
    parsed: dict[int, dict[str, str]] = {}

    def sections_for(number: int) -> dict[str, str] | None:
        if number not in parsed:
            path = CACHE / f"rfc{number}.txt"
            if not path.exists():
                return None
            doc = parse_rfc(number, path.read_text(encoding="utf-8", errors="replace"))
            parsed[number] = {s.number: s.text for s in doc.numbered_sections if s.number}
        return parsed[number]

    for question in golden:
        for ref in question.relevant:
            sections = sections_for(ref.rfc_number)
            if sections is None:
                problems.append(f"{question.id}: RFC{ref.rfc_number} is not in the cache")
                continue
            if ref.section is None:
                continue
            matching = [
                number
                for number in sections
                if number == ref.section or number.startswith(f"{ref.section}.")
            ]
            if not matching:
                problems.append(f"{question.id}: RFC{ref.rfc_number} has no section {ref.section}")

        if question.must_mention:
            corpus_text = " ".join(
                text
                for ref in question.relevant
                if (sections := sections_for(ref.rfc_number)) is not None
                for number, text in sections.items()
                if ref.section is None
                or number == ref.section
                or number.startswith(f"{ref.section}.")
            ).lower()
            for term in question.must_mention:
                if term.lower() not in corpus_text:
                    problems.append(f"{question.id}: {term!r} not found in the labelled sections")

    counts = {kind.value: len(golden.by_type(kind)) for kind in QuestionType}
    print(f"golden set: {len(golden)} questions {counts}")
    print(f"scoreable (retrieval): {len(golden.scoreable)}")
    print(f"RFCs referenced: {len(golden.rfc_numbers())}")

    if problems:
        print(f"\n{len(problems)} problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("\nall labelled sections exist and contain their required terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
