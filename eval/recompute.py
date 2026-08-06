"""Recompute metrics over stored runs.

Every run records its per-question relevance array, not just the aggregates. That
means a corrected metric can be applied to historical runs without re-embedding a
corpus or re-querying a model - the retrieval itself never changed, only the
arithmetic over it.

This exists because ``ndcg_at_k`` had a real defect: it built the ideal ranking from
the labelled section count, but subsection matching lets one label be satisfied by
several retrieved chunks, so achieved DCG could exceed the "ideal" and nDCG could
exceed 1.0. Rather than withdraw the affected numbers, they are recomputed.

    python -m eval.recompute --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.harness import K_VALUES, RESULTS_DIR
from eval.metrics import hit_rate, mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def recompute_question(question: dict[str, Any]) -> dict[str, Any]:
    relevance = [bool(x) for x in question["relevance"]]
    total_relevant = len(question["relevant"])

    question["recall"] = {
        f"@{n}": round(recall_at_k(relevance, total_relevant, n), 4) for n in K_VALUES
    }
    question["precision"] = {f"@{n}": round(precision_at_k(relevance, n), 4) for n in K_VALUES}
    question["hit"] = {f"@{n}": hit_rate(relevance, n) for n in K_VALUES}
    question["mrr"] = round(reciprocal_rank(relevance), 4)
    question["ndcg_10"] = round(ndcg_at_k(relevance, total_relevant, 10), 4)
    return question


def _aggregate(questions: list[dict[str, Any]]) -> dict[str, float]:
    if not questions:
        return {}
    agg: dict[str, float] = {}
    for n in K_VALUES:
        agg[f"recall@{n}"] = round(mean([q["recall"][f"@{n}"] for q in questions]), 4)
        agg[f"hit@{n}"] = round(mean([q["hit"][f"@{n}"] for q in questions]), 4)
    agg["precision@5"] = round(mean([q["precision"]["@5"] for q in questions]), 4)
    agg["mrr"] = round(mean([q["mrr"] for q in questions]), 4)
    agg["ndcg@10"] = round(mean([q["ndcg_10"] for q in questions]), 4)
    # Latency is a property of the original run, not of the metric, so it is carried
    # through untouched.
    return agg


def recompute_run(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    before = dict(payload.get("aggregate", {}))
    questions = [recompute_question(q) for q in payload["questions"]]

    latency = payload.get("aggregate", {}).get("p50_latency_ms")
    aggregate = _aggregate(questions)
    if latency is not None:
        aggregate["p50_latency_ms"] = latency
    aggregate["questions"] = len(questions)
    payload["aggregate"] = aggregate

    types = {q["type"] for q in questions}
    payload["by_type"] = {
        kind: _aggregate([q for q in questions if q["type"] == kind]) for kind in sorted(types)
    }
    return payload, before


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--write", action="store_true", help="update the files in place")
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.results_dir.glob("*.json"))
    if not paths:
        print(f"no result files in {args.results_dir}")
        return 0

    print(f"{'run':<44}{'ndcg@10 was':>13}{'now':>9}{'mrr':>9}")
    print("-" * 75)
    changed = 0
    for path in paths:
        payload = json.loads(path.read_text())
        if "questions" not in payload:
            continue
        payload, before = recompute_run(payload)
        after = payload["aggregate"]
        was, now = before.get("ndcg@10", 0.0), after.get("ndcg@10", 0.0)
        if abs(was - now) > 1e-9:
            changed += 1
        flag = " *" if abs(was - now) > 1e-9 else "  "
        print(f"{path.stem[:42]:<44}{was:>13.4f}{now:>9.4f}{after.get('mrr', 0):>9.4f}{flag}")
        if args.write:
            path.write_text(json.dumps(payload, indent=2))

    print()
    print(f"{changed} of {len(paths)} runs had a corrected nDCG")
    if not args.write:
        print("(dry run - pass --write to update the files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
