// Load profile for POST /api/ask.
//
// Different in kind from search, and the numbers are not comparable. Every request runs
// a multi-round agent loop against a rate-limited third-party model, holds a database
// connection for its whole life, and costs money. So this measures concurrency the
// endpoint is actually expected to survive - a handful of simultaneous readers - rather
// than a throughput ceiling nobody will pay to reach.
//
//   k6 run -e BASE_URL=http://localhost:8000 load/ask.js
//
// Deliberately kept small. At roughly $0.003 a question a careless run is a real bill,
// and the deployed rate limiter will 429 anything more aggressive than this anyway,
// which the script treats as a pass rather than an error: being limited is the limiter
// working.

import http from "k6/http";
import { check } from "k6";
import { Trend, Rate, Counter } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const VUS = Number(__ENV.VUS || 3);
const ITERATIONS = Number(__ENV.ITERATIONS || 12);

const QUESTIONS = [
  "Is the Host header still mandatory?",
  "What must a client do when it receives a 417 response?",
  "Which RFC obsoleted RFC 2616?",
  "What does the exp claim mean in a JWT?",
];

const askLatency = new Trend("sift_ask_latency", true);
const timeToFirstByte = new Trend("sift_ask_ttfb", true);
const rateLimited = new Counter("sift_ask_rate_limited");
const failures = new Rate("sift_ask_failures");

export const options = {
  scenarios: {
    steady: {
      executor: "shared-iterations",
      vus: VUS,
      iterations: ITERATIONS,
      maxDuration: "5m",
    },
  },
  thresholds: {
    // The whole answer, including several retrieval round trips and the model's own
    // generation. Generous because most of it is not ours to control.
    "sift_ask_latency": ["p(95)<45000"],
    "sift_ask_failures": ["rate<0.05"],
  },
};

export default function () {
  const query = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const started = Date.now();

  const response = http.post(`${BASE_URL}/api/ask`, JSON.stringify({ query }), {
    headers: { "Content-Type": "application/json" },
    timeout: "120s",
  });

  if (response.status === 429) {
    rateLimited.add(1);
    failures.add(false);
    return;
  }

  askLatency.add(Date.now() - started);
  // k6 buffers the whole stream, so this is the response's first byte rather than the
  // model's first token. It bounds time-to-first-token from above, no more than that.
  timeToFirstByte.add(response.timings.waiting);

  const ok = check(response, {
    "status is 200": (r) => r.status === 200,
    "stream ended with a done event": (r) => r.body.includes('"type": "done"'),
    "answer carried a citation": (r) => /"citation"/.test(r.body) || /"refused": true/.test(r.body),
  });
  failures.add(!ok);
}
