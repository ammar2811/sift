// Load profile for POST /api/search.
//
// Search is the endpoint that has to survive traffic: it is cheap, unauthenticated, and
// every /api/ask request runs several of them internally through the agent's tools. Its
// tail is the tail everything else inherits.
//
//   k6 run -e BASE_URL=http://localhost:8000 load/search.js
//   k6 run -e BASE_URL=https://sift-api.../ -e STAGE=peak load/search.js
//
// Queries are drawn from the golden set rather than repeated, because a single repeated
// query measures the embedding cache and nothing else. `distinct` and `repeated` below
// exist to measure exactly that difference.

import http from "k6/http";
import { check } from "k6";
import { Trend, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const STAGE = __ENV.STAGE || "default";

// Real questions from eval/golden. Kept inline so the script has no data dependency and
// can be pointed at a deployed environment from anywhere.
const QUERIES = [
  "What information does the HTTP Host header field provide?",
  "What does the HTTP 417 status code mean?",
  "When must a server send a 400 response?",
  "How are DNS label lengths limited?",
  "What is the maximum size of a UDP payload in QUIC?",
  "Which cipher suites are mandatory in TLS 1.3?",
  "What does the Expect header field do?",
  "How does chunked transfer coding terminate?",
  "What is the default port for HTTP over TLS?",
  "When may a cache store a response with an Authorization header?",
  "What does the exp claim mean in a JWT?",
  "Which grant types does OAuth 2.0 define?",
];

const searchLatency = new Trend("sift_search_latency", true);
const failures = new Rate("sift_search_failures");

export const options = {
  scenarios: {
    // A slow ramp rather than a fixed rate: the API scales to zero, and arriving at
    // full load against a cold replica measures the cold start, not the steady state.
    // load/coldstart.js measures that deliberately instead.
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages:
        STAGE === "peak"
          ? [
              { duration: "30s", target: 20 },
              { duration: "1m", target: 20 },
              { duration: "15s", target: 0 },
            ]
          : [
              { duration: "20s", target: 5 },
              { duration: "40s", target: 5 },
              { duration: "10s", target: 0 },
            ],
      gracefulRampDown: "10s",
    },
  },
  thresholds: {
    // Not aspirations. These are the numbers a first local run produced, rounded out,
    // so a regression fails the run instead of being noticed later in a chart.
    "http_req_failed": ["rate<0.01"],
    "sift_search_latency": ["p(95)<2000", "p(99)<4000"],
  },
};

export default function () {
  // Two thirds distinct, one third repeated, so a run reports both the cold path and
  // the cached one rather than an average that hides which is which.
  const repeated = __ITER % 3 === 0;
  const query = repeated ? QUERIES[0] : QUERIES[Math.floor(Math.random() * QUERIES.length)];

  const response = http.post(
    `${BASE_URL}/api/search`,
    JSON.stringify({ query, k: 10, mode: "hybrid" }),
    { headers: { "Content-Type": "application/json" }, tags: { cached: String(repeated) } },
  );

  searchLatency.add(response.timings.duration, { cached: String(repeated) });
  const ok = check(response, {
    "status is 200": (r) => r.status === 200,
    "returned results": (r) => {
      try {
        return JSON.parse(r.body).results.length > 0;
      } catch {
        return false;
      }
    },
  });
  failures.add(!ok);
}
