// Time to first response after the deployment has scaled to zero.
//
// The API runs with minReplicas: 0, which is what makes it free at rest and is also a
// real part of the user experience: the first visitor after a quiet period waits for a
// container to start, an embedding provider to initialise and a connection pool to
// open. That wait appears in no steady-state percentile, and a load test that ignores
// it reports a latency profile nobody experiences.
//
// Scale-to-zero on Container Apps takes several minutes of no traffic, so this is run
// deliberately rather than as part of a suite:
//
//   # wait out the scale-down window first, then
//   k6 run -e BASE_URL=https://sift-api.../ load/coldstart.js
//
// One iteration, one request. Repeating it inside a run would measure a warm replica.

import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";

const coldHealth = new Trend("sift_cold_health", true);
const coldSearch = new Trend("sift_cold_search", true);

export const options = {
  scenarios: {
    once: { executor: "shared-iterations", vus: 1, iterations: 1, maxDuration: "5m" },
  },
};

export default function () {
  // /health is dependency-free, so this isolates container start from everything the
  // application does afterwards.
  const health = http.get(`${BASE_URL}/health`, { timeout: "180s" });
  coldHealth.add(health.timings.duration);
  check(health, { "health answered": (r) => r.status === 200 });

  // The first real query then pays for whatever startup deferred: provider
  // initialisation, the first pool connection, and an unwarmed pgvector index.
  const search = http.post(
    `${BASE_URL}/api/search`,
    JSON.stringify({ query: "What information does the HTTP Host header field provide?", k: 10 }),
    { headers: { "Content-Type": "application/json" }, timeout: "180s" },
  );
  coldSearch.add(search.timings.duration);
  check(search, { "first search answered": (r) => r.status === 200 });

  console.log(
    `cold /health ${health.timings.duration.toFixed(0)}ms | ` +
      `first /api/search ${search.timings.duration.toFixed(0)}ms`,
  );
}
