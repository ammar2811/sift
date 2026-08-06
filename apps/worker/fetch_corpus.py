"""Download RFC texts into a local disk cache.

Ingestion gets re-run many times while tuning chunking, so the RFC Editor should be
hit exactly once per document across the whole project. Everything here is
resumable: rerunning skips files already on disk, so an interrupted download costs
only the documents it had not reached.

Usage:
    python -m apps.worker.fetch_corpus --refresh-index
    python -m apps.worker.fetch_corpus --proposed-since 2020
    python -m apps.worker.fetch_corpus --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from packages.sift_core.rfc_index import (
    INDEX_URL,
    TEXT_URL,
    CorpusSelector,
    RfcMeta,
    load_index,
)

DATA = Path(__file__).resolve().parents[2] / "data"
INDEX_PATH = DATA / "rfc-index.xml"
CACHE = DATA / "rfc-cache"

# Contact details make an unattended crawler identifiable, which is the minimum
# courtesy owed to a free public archive.
USER_AGENT = "sift-rfc-indexer/0.1 (portfolio project; +https://github.com/ammar-siddiqui)"

MAX_CONCURRENCY = 4
MIN_INTERVAL_S = 0.15  # ~6 req/s ceiling across all workers
MAX_ATTEMPTS = 4


class Throttle:
    """Serialises request start times so concurrency never exceeds the rate ceiling."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
                now = loop.time()
            self._next_at = now + self._min_interval


async def _download(
    client: httpx.AsyncClient, meta: RfcMeta, throttle: Throttle, sem: asyncio.Semaphore
) -> tuple[int, str]:
    dest = CACHE / f"rfc{meta.number}.txt"
    if dest.exists() and dest.stat().st_size > 0:
        return meta.number, "cached"

    url = TEXT_URL.format(number=meta.number)
    async with sem:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            await throttle.wait()
            try:
                r = await client.get(url)
                if r.status_code == 404:
                    return meta.number, "missing"
                r.raise_for_status()
            except (httpx.HTTPError, httpx.StreamError) as exc:
                if attempt == MAX_ATTEMPTS:
                    return meta.number, f"failed: {type(exc).__name__}"
                await asyncio.sleep(2**attempt)
                continue

            # Write via a temp file so an interrupted run never leaves a truncated
            # document that a later run would treat as successfully cached.
            tmp = dest.with_suffix(".partial")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return meta.number, "downloaded"
    return meta.number, "failed"


async def fetch_all(metas: list[RfcMeta]) -> dict[str, int]:
    CACHE.mkdir(parents=True, exist_ok=True)
    throttle = Throttle(MIN_INTERVAL_S)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    counts: dict[str, int] = {}

    limits = httpx.Limits(max_connections=MAX_CONCURRENCY)
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=60.0, limits=limits, follow_redirects=True
    ) as client:
        tasks = [_download(client, m, throttle, sem) for m in metas]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            number, outcome = await coro
            key = outcome.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if outcome.startswith(("failed", "missing")):
                print(f"  RFC{number}: {outcome}", file=sys.stderr)
            if i % 100 == 0 or i == len(tasks):
                done = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"[{i}/{len(tasks)}] {done}", flush=True)
    return counts


async def refresh_index() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"fetching {INDEX_URL}")
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=180.0) as client:
        r = await client.get(INDEX_URL)
        r.raise_for_status()
    tmp = INDEX_PATH.with_suffix(".partial")
    tmp.write_bytes(r.content)
    tmp.replace(INDEX_PATH)
    print(f"wrote {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh-index", action="store_true", help="re-download rfc-index.xml first")
    p.add_argument(
        "--proposed-since",
        type=int,
        default=2020,
        help="include non-obsoleted Proposed Standards from this year onward (0 to exclude)",
    )
    p.add_argument("--limit", type=int, default=0, help="cap the number of documents (0 = all)")
    p.add_argument(
        "--rfc",
        type=int,
        nargs="+",
        default=(),
        help="fetch these RFC numbers regardless of the selector (foundational specs "
        "such as TLS 1.3 and OAuth 2.0 predate the Proposed Standard year cutoff)",
    )
    p.add_argument("--dry-run", action="store_true", help="report the selection and exit")
    args = p.parse_args()

    if args.refresh_index or not INDEX_PATH.exists():
        asyncio.run(refresh_index())

    metas = load_index(INDEX_PATH)
    if args.rfc:
        wanted = set(args.rfc)
        selected = [m for m in metas if m.number in wanted]
    else:
        selector = CorpusSelector(proposed_since=args.proposed_since or None)
        selected = selector.apply(metas)
        if args.limit:
            selected = selected[: args.limit]

    pages = sum(m.page_count for m in selected)
    print(
        f"index: {len(metas)} RFCs -> selected {len(selected)} "
        f"({pages:,} pages, ~{pages * 2700 / 1e6:.0f}M chars)"
    )
    if args.dry_run:
        return 0

    counts = asyncio.run(fetch_all(selected))
    print("done:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
