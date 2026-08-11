"""Run the golden set through the agent and record what came back.

The retrieval harness measures whether the right passage can be found. This measures
whether the answer built from it is any good, which is a different question and until
now an unmeasured one.

It runs the whole golden set, unanswerable questions included. Those are the point of
having them: retrieval cannot be scored on a question with no correct passage, but an
*answer* to one can be scored exactly - either the system declined or it made something
up. Abstention has been claimed by this project since the golden set was written and
has never been measured.

Costs real money, roughly $0.25 a run against gpt-4.1-mini plus the judge, so it is a
command you run deliberately rather than a test.

    python -m eval.answer_harness --label baseline
    python -m eval.answer_harness --no-judge --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.answer_metrics import (
    citation_precision,
    citation_recall,
    is_grounded,
    mention_coverage,
    uncited_rate,
)
from eval.golden_set import GOLDEN_DIR, GoldenSet, Question, QuestionType, load_golden_set
from eval.harness import RESULTS_DIR, _git_sha
from eval.judge import Judge, Judgement, Verdict, build_judge
from eval.metrics import mean
from packages.sift_core import db
from packages.sift_core.agent import Agent
from packages.sift_core.config import get_settings
from packages.sift_core.providers import build_chat_model, build_embedding_provider
from packages.sift_core.tools import ToolContext

logger = logging.getLogger("sift.answer_harness")


@dataclass(slots=True)
class AnswerResult:
    id: str
    type: str
    question: str
    answer: str
    citations: list[str]
    refused: bool
    hit_limit: bool
    trajectory: list[dict[str, Any]]
    latency_s: float
    usage: dict[str, Any]
    # Programmatic, computed without a model.
    grounded: bool = False
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    mention_coverage: float = 0.0
    # Judged.
    verdict: str = Verdict.UNGRADED.value
    judge_reason: str = ""
    # Set when the question could not be answered at all - a rate limit, a timeout. Such
    # a question is excluded from every quality metric: it measures the run, not the
    # system, and averaging it in would read as a quality regression.
    error: str | None = None


@dataclass(slots=True)
class AnswerRun:
    label: str
    created_at: str
    config: dict[str, Any]
    aggregate: dict[str, Any]
    by_type: dict[str, dict[str, Any]]
    questions: list[AnswerResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)


def answer_question(agent: Agent, provider: Any, question: Question) -> AnswerResult:
    """Answer one question on its own connection, so a run can be parallel.

    A question that fails outright is recorded rather than raised. A run costs money and
    takes minutes, and losing every completed answer because the provider rate-limited
    the fifty-first question is a bad trade - the failure is more useful sitting in the
    output where it can be counted.
    """
    started = time.perf_counter()
    try:
        with db.connection() as conn:
            version_id = db.active_version_id(conn)
            if version_id is None:
                raise SystemExit("no active corpus version - ingest and activate first")
            ctx = ToolContext(conn, version_id, provider.embed_query)
            result = agent.run(question.question, ctx)
    except SystemExit:
        raise
    except Exception as exc:
        logger.warning("%s failed: %s", question.id, exc)
        return AnswerResult(
            id=question.id,
            type=question.type.value,
            question=question.question,
            answer="",
            citations=[],
            refused=False,
            hit_limit=False,
            trajectory=[],
            latency_s=round(time.perf_counter() - started, 2),
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tool_calls": 0,
                "rounds": 0,
                "cost_usd": 0.0,
            },
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    citations = list(result.citations)
    labelled = list(question.relevant)
    return AnswerResult(
        id=question.id,
        type=question.type.value,
        question=question.question,
        answer=result.answer,
        citations=citations,
        refused=result.refused,
        hit_limit=result.hit_limit,
        trajectory=result.trajectory,
        latency_s=round(time.perf_counter() - started, 2),
        usage={
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "tool_calls": result.usage.tool_calls,
            "rounds": result.usage.rounds,
            "cost_usd": round(result.usage.cost_usd(), 6),
        },
        grounded=is_grounded(citations, labelled) if labelled else False,
        citation_precision=round(citation_precision(citations, labelled), 4) if labelled else 0.0,
        citation_recall=round(citation_recall(citations, labelled), 4) if labelled else 0.0,
        mention_coverage=round(mention_coverage(result.answer, question.must_mention), 4),
    )


def _aggregate(all_results: list[AnswerResult]) -> dict[str, Any]:
    if not all_results:
        return {}

    failed = [r for r in all_results if r.error]
    results = [r for r in all_results if not r.error]
    if not results:
        return {"questions": len(all_results), "errors": len(failed)}

    answerable = [r for r in results if r.type != QuestionType.UNANSWERABLE.value]
    unanswerable = [r for r in results if r.type == QuestionType.UNANSWERABLE.value]
    graded = [r for r in results if r.verdict != Verdict.UNGRADED.value]

    agg: dict[str, Any] = {"questions": len(results)}
    if failed:
        agg["errors"] = len(failed)
    if graded:
        credit = {Verdict.CORRECT.value: 1.0, Verdict.PARTIAL.value: 0.5}
        agg["judged_correct"] = round(
            mean([1.0 if r.verdict == Verdict.CORRECT.value else 0.0 for r in graded]), 4
        )
        agg["judged_score"] = round(mean([credit.get(r.verdict, 0.0) for r in graded]), 4)
        agg["graded"] = len(graded)
        agg["ungraded"] = len(results) - len(graded)

    if answerable:
        agg["grounded"] = round(mean([1.0 if r.grounded else 0.0 for r in answerable]), 4)
        agg["citation_precision"] = round(mean([r.citation_precision for r in answerable]), 4)
        agg["citation_recall"] = round(mean([r.citation_recall for r in answerable]), 4)
        agg["mention_coverage"] = round(mean([r.mention_coverage for r in answerable]), 4)

    if unanswerable:
        agg["abstention_rate"] = round(mean([1.0 if r.refused else 0.0 for r in unanswerable]), 4)

    # Answers that assert without citing, which is the failure the redo path exists to
    # catch. Measured over answerable questions: declining to answer is not this failure.
    agg["uncited_assertions"] = round(
        uncited_rate([(bool(r.citations), r.refused) for r in answerable]), 4
    )
    agg["hit_limit_rate"] = round(mean([1.0 if r.hit_limit else 0.0 for r in results]), 4)
    agg["p50_latency_s"] = round(statistics.median([r.latency_s for r in results]), 2)
    agg["mean_tool_calls"] = round(mean([float(r.usage["tool_calls"]) for r in results]), 2)
    agg["mean_cost_usd"] = round(mean([float(r.usage["cost_usd"]) for r in results]), 6)
    agg["total_cost_usd"] = round(sum(float(r.usage["cost_usd"]) for r in results), 4)
    return agg


def run_answer_evaluation(
    golden: GoldenSet,
    *,
    label: str,
    judge: Judge | None,
    concurrency: int,
    limit: int | None = None,
) -> AnswerRun:
    settings = get_settings()
    provider = build_embedding_provider(settings)
    chat = build_chat_model(settings)

    # One agent across threads: it holds no per-question state, and the OpenAI client
    # underneath it is thread-safe and pools its connections.
    agent = Agent(
        chat.client,
        chat.deployment,
        max_depth=settings.agent_max_depth,
        max_tool_calls=settings.agent_max_tool_calls,
        max_completion_tokens=settings.agent_max_completion_tokens,
        cost_prompt_per_m=settings.chat_cost_prompt_per_m,
        cost_completion_per_m=settings.chat_cost_completion_per_m,
    )
    questions = list(golden.questions)[: limit or None]

    print(f"answering {len(questions)} questions at concurrency {concurrency}…", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda q: answer_question(agent, provider, q), questions))

    if judge is not None:
        print("judging…", file=sys.stderr)
        lookup = {q.id: q for q in questions}

        def grade(result: AnswerResult) -> Judgement:
            return judge.grade(result.question, lookup[result.id].reference_answer, result.answer)

        gradeable = [r for r in results if not r.error]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for result, judgement in zip(gradeable, pool.map(grade, gradeable), strict=True):
                result.verdict = judgement.verdict.value
                result.judge_reason = judgement.reason

    with db.connection() as conn:
        version_id = db.active_version_id(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM corpus_versions WHERE id = %s", (version_id,))
            version = dict(cur.fetchone() or {})
        stats = db.corpus_stats(conn, version_id) if version_id else {}

    by_type = {
        kind.value: _aggregate([r for r in results if r.type == kind.value])
        for kind in QuestionType
        if any(r.type == kind.value for r in results)
    }

    config = {
        "chat_deployment": chat.deployment,
        "judge_deployment": settings.azure_reasoning_deployment if judge else None,
        "agent_max_depth": settings.agent_max_depth,
        "agent_max_tool_calls": settings.agent_max_tool_calls,
        "agent_max_completion_tokens": settings.agent_max_completion_tokens,
        "embedding_provider": provider.name,
        "dimensions": provider.dimensions,
        "corpus_fingerprint": version.get("fingerprint"),
        "corpus": dict(stats),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
    }

    return AnswerRun(
        label=label,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config=config,
        aggregate=_aggregate(results),
        by_type=by_type,
        questions=results,
    )


def render(run: AnswerRun) -> str:
    agg = run.aggregate
    lines = [
        f"{run.label}  ({run.created_at})",
        f"  corpus     : {run.config['corpus'].get('documents')} docs, "
        f"{run.config['corpus'].get('chunks')} chunks",
        f"  answering  : {run.config['chat_deployment']}",
        f"  judging    : {run.config['judge_deployment'] or 'not judged'}",
        "",
        f"  {'metric':<22}{'value':>9}",
        f"  {'-' * 31}",
    ]
    for key in (
        "judged_correct",
        "judged_score",
        "grounded",
        "citation_recall",
        "citation_precision",
        "mention_coverage",
        "abstention_rate",
        "uncited_assertions",
        "hit_limit_rate",
    ):
        if key in agg:
            lines.append(f"  {key:<22}{agg[key]:>9.4f}")

    if agg.get("errors"):
        lines.append(f"  {'errors':<22}{agg['errors']:>9}  (excluded from the metrics above)")

    lines += [
        "",
        f"  {'p50 latency':<22}{agg.get('p50_latency_s', 0):>8.2f}s",
        f"  {'mean tool calls':<22}{agg.get('mean_tool_calls', 0):>9.2f}",
        f"  {'cost / question':<22}{agg.get('mean_cost_usd', 0):>9.4f}",
        f"  {'cost / run':<22}{agg.get('total_cost_usd', 0):>9.4f}",
        "",
        f"  {'by type':<16}{'judged':>9}{'grounded':>10}{'abstain':>9}{'n':>5}",
        f"  {'-' * 49}",
    ]

    def cell(by: dict[str, Any], key: str, width: int) -> str:
        """A metric that does not apply to a question type gets a dash, not a nan.

        Abstention is undefined for answerable questions and grounding is undefined for
        unanswerable ones, and printing nan invites someone to average it later.
        """
        return f"{by[key]:>{width}.4f}" if key in by else f"{'-':>{width}}"

    for kind, by in run.by_type.items():
        lines.append(
            f"  {kind:<16}{cell(by, 'judged_correct', 9)}{cell(by, 'grounded', 10)}"
            f"{cell(by, 'abstention_rate', 9)}{int(by.get('questions', 0)):>5}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", default="answers")
    p.add_argument("--golden-dir", type=Path, default=GOLDEN_DIR)
    p.add_argument("--no-judge", action="store_true", help="programmatic metrics only")
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel questions; each holds a database connection, so keep under the pool size",
    )
    p.add_argument("--limit", type=int, default=None, help="answer only the first N questions")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)

    golden = load_golden_set(args.golden_dir)
    run = run_answer_evaluation(
        golden,
        label=args.label,
        judge=None if args.no_judge else build_judge(),
        concurrency=max(1, args.concurrency),
        limit=args.limit,
    )
    print(render(run))

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = args.out or RESULTS_DIR / f"{args.label}-{stamp}.json"
        path.write_text(run.to_json())
        print(f"\nwrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
