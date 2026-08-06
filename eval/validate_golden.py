"""Check the golden set against the corpus.

A labelled section that does not exist, or that does not contain the answer, silently
corrupts every metric computed from it - recall would be measured against an
unreachable target. This runs over the parsed corpus rather than the database, so it
works before anything has been ingested.

    python -m eval.validate_golden
"""

from __future__ import annotations

import sys
from pathlib import Path

from eval.golden_set import GOLDEN_DIR, QuestionType, load_golden_set
from packages.sift_core.rfc_parser import parse_rfc

CACHE = Path(__file__).resolve().parents[1] / "data" / "rfc-cache"


def main() -> int:
    golden = load_golden_set(GOLDEN_DIR)
    problems: list[str] = []
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
