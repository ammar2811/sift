# Load results

k6 profiles for the two endpoints that serve users, plus the cold start that neither of
them measures. Run them with the Docker image so there is nothing to install:

```
docker run --rm --network host -v "$PWD/load:/scripts" \
  -e BASE_URL=http://localhost:8000 grafana/k6:latest run /scripts/search.js
```

**Where these numbers came from, before anything else.** Everything below was measured
against a local API and a local Postgres, with embeddings served by Azure OpenAI over
the public internet. It is not the deployed environment: the deployed database is a
Burstable B1ms with a fraction of this machine's memory and IO, and the deployed API
scales to zero. **The percentiles here are not the percentiles a user of the deployed
site would see, and the cold-start profile has not been run at all** - it needs the
scale-to-zero behaviour that only exists on Container Apps. Treat these as a floor and
as a regression baseline, not as production numbers.

## Search

`load/search.js`, ramping to 5 concurrent users over 70 seconds, 264 requests, hybrid
mode at k=10, Redis enabled.

| | |
|---|---|
| p50 | 1.19 s |
| p90 | 1.53 s |
| p95 | 1.74 s |
| p99 | 1.90 s |
| min | 229 ms |
| max | 2.29 s |
| throughput | 3.76 req/s |
| failures | 0 of 264 |

The gap between the minimum and the median is the whole story: 229 ms is a query whose
embedding was cached, and 1.19 s is one that was not. Roughly a second of the median is
a round trip to Azure OpenAI in East US to embed the query, before Postgres is touched
at all. The retrieval eval reports a p50 of 452 ms for the same work, because it
measures the database query and this measures the request.

That also means the thresholds in the script guard the wrong thing if the embedding
provider changes. They are set where a regression fails the run, not where the system
is comfortable.

## What Redis is actually worth

The obvious experiment - run the load test with and without Redis - shows almost no
difference, because `CachedEmbeddings` is two tiers: an in-process dictionary in front
of Redis. Within a single run, in a single process, the dictionary serves every repeat
and Redis is never consulted.

The value is elsewhere. The API runs with `minReplicas: 0`, so the process does not
survive a quiet period. Timing the same query around a restart:

| | with Redis | without Redis |
|---|---|---|
| first request | 5.95 s* | 0.93 s |
| repeat, same process | 0.24 s | 0.29 s |
| **repeat, after a restart** | **0.30 s** | **0.92 s** |

\* the first request of all, which also pays for the first HTTPS connection to Azure and
the first pool connection. It is not a cache measurement and is included only so the
run is reported whole.

Same process, Redis changes nothing. Across a restart it is the difference between
300 ms and 920 ms, and on a deployment that scales to zero, "across a restart" is the
common case rather than the exception. **Redis stays on.** That is the decision its
Bicep parameter was waiting for.

## Ask

`load/ask.js`, 3 concurrent users, 12 questions, rate limiter disabled for the run.

| | |
|---|---|
| p50 | 3.81 s |
| p90 | 10.80 s |
| p95 | 11.23 s |
| max | 11.43 s |
| failures | 0 of 12 |
| checks | 36 of 36 passed |

Every response ended with a `done` event and carried either a citation or a refusal,
which is the same guarantee the answer evaluation measures, checked here under
concurrency instead.

The spread is wide because the work is variable: a question answered from one search
finishes in about two seconds, and one that walks a supersession chain and then reads a
section runs four tool rounds. The measured mean is 3.1 tool calls per question. This is
not a throughput number and should not be read as one - 3 concurrent users is roughly
what the rate limiter and the connection pool permit by design.

`sift_ask_ttfb` reports 2-5 ms, which is the response headers rather than the first
token: k6 buffers a streamed body, so it cannot see the stream. Time to first *token* is
a real metric this cannot measure, and the number to beat is the 985 ms recorded for
gpt-4.1-mini in `config.py`.

## Known limitations

- Not measured against the deployed environment. Everything above is a local floor.
- `load/coldstart.js` has never been run. Scale-from-zero is a real part of the deployed
  experience and is currently unquantified.
- Postgres here is local and warm. The B1ms has 2 GiB of memory against a 129,109-chunk
  HNSW index, and behaves differently under concurrency.
- No sustained soak. The longest run is 70 seconds, which is long enough to find a
  latency profile and far too short to find a leak.
- The ask profile is small enough to be cheap ($0.04 a run) and therefore too small for
  its p95 to mean much: 12 samples.
- One region, one client, one machine. Network conditions are not controlled for.
