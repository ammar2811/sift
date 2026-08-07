"""Run the golden set against retrieval and record the numbers.

Retrieval quality is measurable without a generation model: the labels name the
sections that should come back, so recall, MRR and nDCG are computable from the
retriever alone. That is deliberate - it means the optimization sweep can run, and a
baseline can be recorded, before any LLM is wired up.

Every run writes a timestamped JSON file containing both the metrics and the exact
configuration that produced them, so a README table entry can always be traced back
to a reproducible run.

    python -m eval.harness --mode hybrid --label baseline
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.golden_set import GOLDEN_DIR, GoldenSet, Question, QuestionType, load_golden_set
from eval.metrics import hit_rate, mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from packages.sift_core import db
from packages.sift_core.config import get_settings
from packages.sift_core.providers import build_embedding_provider
from packages.sift_core.retrieval import (
    DEFAULT_KEYWORD_WEIGHT,
    KeywordSemantics,
    SearchMode,
    search,
)

RESULTS_DIR = Path(__file__).parent / "results"
K_VALUES = (1, 3, 5, 10)


@dataclass(slots=True)
class QuestionResult:
    id: str
    type: str
    question: str
    relevant: list[str]
    retrieved: list[str]
    relevance: list[bool]
    latency_ms: float
    recall: dict[str, float] = field(default_factory=dict)
    precision: dict[str, float] = field(default_factory=dict)
    hit: dict[str, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_10: float = 0.0


@dataclass(slots=True)
class RunResult:
    label: str
    created_at: str
    config: dict[str, Any]
    aggregate: dict[str, float]
    by_type: dict[str, dict[str, float]]
    questions: list[QuestionResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def evaluate_question(
    conn: Any,
    version_id: int,
    question: Question,
    provider: Any,
    *,
    mode: SearchMode,
    k: int,
    candidates: int,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    keyword_semantics: KeywordSemantics = KeywordSemantics.IDF,
) -> QuestionResult:
    started = time.perf_counter()
    embedding = (
        provider.embed_query(question.question)
        if mode in (SearchMode.DENSE, SearchMode.HYBRID)
        else None
    )
    hits = search(
        conn,
        version_id,
        query=question.question,
        embedding=embedding,
        mode=mode,
        k=k,
        candidates=candidates,
        keyword_weight=keyword_weight,
        keyword_semantics=keyword_semantics,
    )
    latency_ms = (time.perf_counter() - started) * 1000

    relevance = [question.is_relevant(h.rfc_number, h.section_number) for h in hits]
    total_relevant = len(question.relevant)

    return QuestionResult(
        id=question.id,
        type=question.type.value,
        question=question.question,
        relevant=[str(r) for r in question.relevant],
        retrieved=[f"{h.rfc_number}:{h.section_number}" for h in hits],
        relevance=relevance,
        latency_ms=round(latency_ms, 1),
        recall={f"@{n}": round(recall_at_k(relevance, total_relevant, n), 4) for n in K_VALUES},
        precision={f"@{n}": round(precision_at_k(relevance, n), 4) for n in K_VALUES},
        hit={f"@{n}": hit_rate(relevance, n) for n in K_VALUES},
        mrr=round(reciprocal_rank(relevance), 4),
        ndcg_10=round(ndcg_at_k(relevance, total_relevant, 10), 4),
    )


def _aggregate(results: list[QuestionResult]) -> dict[str, float]:
    if not results:
        return {}
    agg: dict[str, float] = {}
    for n in K_VALUES:
        agg[f"recall@{n}"] = round(mean([r.recall[f"@{n}"] for r in results]), 4)
        agg[f"hit@{n}"] = round(mean([r.hit[f"@{n}"] for r in results]), 4)
    agg["precision@5"] = round(mean([r.precision["@5"] for r in results]), 4)
    agg["mrr"] = round(mean([r.mrr for r in results]), 4)
    agg["ndcg@10"] = round(mean([r.ndcg_10 for r in results]), 4)
    agg["p50_latency_ms"] = round(statistics.median([r.latency_ms for r in results]), 1)
    agg["questions"] = len(results)
    return agg


def run_evaluation(
    golden: GoldenSet,
    *,
    mode: SearchMode,
    k: int,
    candidates: int,
    label: str,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    keyword_semantics: KeywordSemantics = KeywordSemantics.IDF,
) -> RunResult:
    settings = get_settings()
    provider = build_embedding_provider(settings)

    with db.connection() as conn:
        version_id = db.active_version_id(conn)
        if version_id is None:
            raise SystemExit(
                "no active corpus version - ingest first: "
                "python -m apps.worker.ingest --sweep-corpus --activate"
            )
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM corpus_versions WHERE id = %s", (version_id,))
            version = dict(cur.fetchone() or {})
        stats = db.corpus_stats(conn, version_id)

        results = [
            evaluate_question(
                conn,
                version_id,
                q,
                provider,
                mode=mode,
                k=k,
                candidates=candidates,
                keyword_weight=keyword_weight,
                keyword_semantics=keyword_semantics,
            )
            for q in golden.scoreable
        ]

    by_type = {
        kind.value: _aggregate([r for r in results if r.type == kind.value])
        for kind in QuestionType
        if any(r.type == kind.value for r in results)
    }

    config = {
        "mode": mode.value,
        "k": k,
        "candidates": candidates,
        "keyword_weight": keyword_weight,
        "keyword_semantics": keyword_semantics.value,
        "embedding_provider": provider.name,
        "dimensions": provider.dimensions,
        "chunk_config": version.get("chunk_config"),
        "corpus_fingerprint": version.get("fingerprint"),
        "corpus": {k2: v for k2, v in stats.items()},
        "git_sha": _git_sha(),
        "python": platform.python_version(),
    }

    return RunResult(
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        aggregate=_aggregate(results),
        by_type=by_type,
        questions=results,
    )


def render(run: RunResult) -> str:
    lines = [
        f"{run.label}  ({run.created_at})",
        f"  corpus     : {run.config['corpus'].get('documents')} docs, "
        f"{run.config['corpus'].get('chunks')} chunks",
        f"  retrieval  : mode={run.config['mode']} k={run.config['k']} "
        f"candidates={run.config['candidates']}",
        f"  embeddings : {run.config['embedding_provider']} ({run.config['dimensions']}d)",
        "",
        f"  {'metric':<16}{'value':>8}",
        f"  {'-' * 24}",
    ]
    for key in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "precision@5"):
        lines.append(f"  {key:<16}{run.aggregate.get(key, 0):>8.4f}")
    lines.append(f"  {'p50 latency':<16}{run.aggregate.get('p50_latency_ms', 0):>7.1f}ms")
    lines.append("")
    lines.append(f"  {'by type':<16}{'recall@5':>10}{'mrr':>8}{'n':>5}")
    lines.append(f"  {'-' * 39}")
    for kind, agg in run.by_type.items():
        lines.append(
            f"  {kind:<16}{agg.get('recall@5', 0):>10.4f}{agg.get('mrr', 0):>8.4f}"
            f"{int(agg.get('questions', 0)):>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="hybrid", choices=[m.value for m in SearchMode])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=50)
    p.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)
    p.add_argument(
        "--keyword-semantics", default="idf", choices=[s.value for s in KeywordSemantics]
    )
    p.add_argument("--label", default=None, help="name for this run (default: mode)")
    p.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    p.add_argument("--out", type=Path, default=None, help="explicit output path")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    golden = load_golden_set(args.golden_dir)
    label = args.label or args.mode
    run = run_evaluation(
        golden,
        mode=SearchMode(args.mode),
        k=args.k,
        candidates=args.candidates,
        label=label,
        keyword_weight=args.keyword_weight,
        keyword_semantics=KeywordSemantics(args.keyword_semantics),
    )
    print(render(run))

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = args.out or RESULTS_DIR / f"{label}-{stamp}.json"
        path.write_text(run.to_json())
        print(f"\nwrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
