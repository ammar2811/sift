"""Which documents the corpus itself depends on.

Corpus selection keeps what is currently in force, plus Proposed Standards from 2020
onward. That year cutoff is a size control, and it has a systematic blind spot: in the
IETF a specification can stay at Proposed Standard permanently, so foundational
documents never advance out of the excluded band. OAuth 2.0 (2012), JWT (2015) and
WebSocket (2011) are all still Proposed Standards and were all missing, while the
corpus that excluded them cited them constantly.

The fix is to let the corpus vote. Counting how many included documents reference each
excluded one measures dependence directly, using nothing but the corpus text, and
answers the question the year cutoff was a poor proxy for: is this document load
bearing for the material already indexed?

    python -m packages.sift_core.citations --min-citations 10
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from packages.sift_core.rfc_index import RfcMeta

DATA = Path(__file__).resolve().parents[2] / "data"
# Computed once and committed, so ingestion does not re-scan the whole corpus and so
# the selection is reproducible - the corpus a result was measured on is a fact about
# that result, not something to recompute later and hope it matched.
MANIFEST = DATA / "load-bearing.json"

# "RFC 2119", "RFC2119" and the line-wrapped "RFC\n   2119" all appear in practice.
_REFERENCE = re.compile(rb"RFC\s*(\d{3,4})")

# Cited by nearly everything for procedural reasons rather than technical ones: these
# define the boilerplate a document carries, not anything a reader would ask about.
# RFC 7841 is cited by 952 of 1,449 documents and is pure noise by this measure.
BOILERPLATE = frozenset({2223, 5741, 7322, 7841, 8792})

# One included document in 150 is a deliberately low bar. It is the loosest threshold
# considered, and it is the one that admits WebSocket at 11 citations - stated plainly
# because the alternative reading is that the threshold was fitted to the evaluation
# set. The effect on retrieval is measured either way, regressions included.
DEFAULT_MIN_CITATIONS = 10


def count_citations(numbers: Iterable[int], cache: Path) -> Counter[int]:
    """How many documents cite each RFC, counting each citing document once.

    Occurrences are deliberately not counted: a specification that mentions RFC 8446
    forty times is one document depending on it, and counting mentions would let a
    single verbose document manufacture importance.
    """
    counts: Counter[int] = Counter()
    for number in numbers:
        path = cache / f"rfc{number}.txt"
        if not path.exists():
            continue
        cited = {int(m) for m in _REFERENCE.findall(path.read_bytes())}
        cited.discard(number)  # self-references in the boilerplate header
        counts.update(cited)
    return counts


def load_bearing(
    metas: Iterable[RfcMeta],
    selected: frozenset[int] | set[int],
    counts: Counter[int],
    *,
    min_citations: int = DEFAULT_MIN_CITATIONS,
) -> frozenset[int]:
    """Excluded documents the corpus leans on hard enough to be worth indexing.

    Superseded documents stay out however heavily cited: citations accumulate over a
    document's whole life, so an obsoleted specification keeps the score it earned
    while it was current. Following them would rebuild the corpus out of history.
    """
    by_number = {m.number: m for m in metas}
    return frozenset(
        number
        for number, cited_by in counts.items()
        if cited_by >= min_citations
        and number not in selected
        and number not in BOILERPLATE
        and (meta := by_number.get(number)) is not None
        and meta.is_current
        and meta.has_text
    )


def load_manifest(path: Path = MANIFEST) -> frozenset[int]:
    """The saved selection, or empty if it has never been computed."""
    if not path.exists():
        return frozenset()
    payload = json.loads(path.read_text())
    return frozenset(int(n) for n in payload["numbers"])


def _main() -> int:
    import argparse

    from packages.sift_core.rfc_index import CorpusSelector, load_index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-citations", type=int, default=DEFAULT_MIN_CITATIONS)
    parser.add_argument("--index", type=Path, default=DATA / "rfc-index.xml")
    parser.add_argument("--cache", type=Path, default=DATA / "rfc-cache")
    parser.add_argument("--out", type=Path, default=MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    metas = load_index(args.index)
    by_number = {m.number: m for m in metas}
    # Deliberately the *unexpanded* selection: the vote is cast by the documents the
    # year cutoff already admitted, so the result cannot bootstrap off itself.
    base = {m.number for m in CorpusSelector(include=frozenset()).apply(metas)}
    counts = count_citations(base, args.cache)
    chosen = load_bearing(metas, base, counts, min_citations=args.min_citations)

    print(f"base corpus {len(base)} documents; {len(chosen)} load-bearing additions")
    pages = sum(by_number[n].page_count for n in chosen)
    print(f"adds {pages:,} pages (~{pages * 2700 / 1e6:.0f}M chars)\n")
    for number in sorted(chosen, key=lambda n: -counts[n])[:15]:
        meta = by_number[number]
        print(f"  {counts[number]:>4} cites  RFC {number:<5} {meta.title[:52]}")
    if len(chosen) > 15:
        print(f"  ... and {len(chosen) - 15} more")

    if args.dry_run:
        return 0
    args.out.write_text(
        json.dumps(
            {
                "min_citations": args.min_citations,
                "base_corpus": len(base),
                "numbers": sorted(chosen),
                "citations": {str(n): counts[n] for n in sorted(chosen)},
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
