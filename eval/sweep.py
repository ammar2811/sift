"""Run the golden set across a range of configurations and print a comparison table.

One variable at a time. Each run writes its own results file, and the table this
prints is the one that belongs in the README - including the rows where a change made
things worse, because a table where everything helped reads as invented.

    python -m eval.sweep keyword-weight
    python -m eval.sweep mode
    python -m eval.sweep candidates
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from eval.golden_set import load_golden_set
from eval.harness import RESULTS_DIR, RunResult, run_evaluation
from packages.sift_core.retrieval import KeywordSemantics, SearchMode

COLUMNS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "p50_latency_ms")


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    kwargs: dict[str, Any]


def keyword_weight_variants() -> Iterator[Variant]:
    """How much the weaker keyword retriever should count in the fusion.

    0.0 is dense-only through the hybrid code path, which makes it a clean control.
    """
    for weight in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        yield Variant(f"kw_weight={weight}", {"keyword_weight": weight})


def mode_variants() -> Iterator[Variant]:
    for mode in (SearchMode.KEYWORD, SearchMode.DENSE, SearchMode.HYBRID):
        yield Variant(f"mode={mode.value}", {"mode": mode})


def semantics_variants() -> Iterator[Variant]:
    for semantics in (KeywordSemantics.ALL, KeywordSemantics.ANY, KeywordSemantics.IDF):
        yield Variant(f"tsquery={semantics.value}", {"keyword_semantics": semantics})


def candidate_variants() -> Iterator[Variant]:
    for candidates in (10, 25, 50, 100, 200):
        yield Variant(f"candidates={candidates}", {"candidates": candidates})


SWEEPS = {
    "keyword-weight": keyword_weight_variants,
    "mode": mode_variants,
    "semantics": semantics_variants,
    "candidates": candidate_variants,
}


def render_table(runs: list[tuple[str, RunResult]]) -> str:
    header = f"| {'configuration':<22} | " + " | ".join(f"{c:>9}" for c in COLUMNS)
    header += f" | {'xdoc r@5':>9} |"
    rule = "|" + "-" * 24 + "|" + "".join("-" * 12 + "|" for _ in COLUMNS) + "-" * 11 + "|"
    lines = [header, rule]

    best = max(runs, key=lambda r: r[1].aggregate.get("recall@5", 0))[0] if runs else ""
    for label, run in runs:
        marker = " *" if label == best else "  "
        cells = " | ".join(f"{run.aggregate.get(c, 0):>9.4f}" for c in COLUMNS)
        xdoc = run.by_type.get("cross_document", {}).get("recall@5", 0)
        lines.append(f"| {label:<20}{marker} | {cells} | {xdoc:>9.4f} |")
    lines.append("")
    lines.append("* best recall@5")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", choices=sorted(SWEEPS))
    parser.add_argument("--mode", default="hybrid", choices=[m.value for m in SearchMode])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    golden = load_golden_set()
    runs: list[tuple[str, RunResult]] = []

    for variant in SWEEPS[args.sweep]():
        kwargs: dict[str, Any] = {
            "mode": SearchMode(args.mode),
            "k": args.k,
            "candidates": 50,
            "label": f"{args.sweep}:{variant.label}",
        }
        kwargs.update(variant.kwargs)
        run = run_evaluation(golden, **kwargs)
        runs.append((variant.label, run))
        if not args.no_save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            safe = variant.label.replace("=", "-").replace(".", "_")
            (RESULTS_DIR / f"sweep-{args.sweep}-{safe}.json").write_text(run.to_json())

    print()
    print(render_table(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
