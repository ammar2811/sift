/**
 * Automated accessibility audit.
 *
 * The project claims WCAG 2.1 AA, and eslint-plugin-jsx-a11y already fails the build on
 * what it can see. It sees JSX in isolation: it cannot tell whether the rendered tree
 * has a heading order, whether a live region ends up nested inside another, or whether
 * an element that gets a role at runtime also gets the attributes that role requires.
 * axe-core inspects the tree the browser would build, so it catches the second class.
 *
 * It runs here rather than as a separate browser-driven audit for one reason: a check
 * that lives in CI is run on every commit, and a check someone remembers to run before a
 * demo is run once. The trade is that jsdom performs no layout, so axe's colour-contrast
 * rule cannot evaluate and is disabled. Contrast is verified separately and
 * deterministically in `contrast.test.ts`, from the same tokens the stylesheet ships.
 */

import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { AskEvent, Citation, ReadinessResponse } from "./lib/api";

const hostChunk: Citation = {
  chunk_id: 1,
  rfc_number: 9110,
  title: "HTTP Semantics",
  section_number: "7.2",
  section_title: "Host and :authority",
  text: 'The "Host" header field in a request provides the host and port information.',
  citation: "RFC 9110 Section 7.2",
  source_url: "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2",
  score: 0.0324,
  dense_rank: 1,
  keyword_rank: 3,
  is_current: false,
  obsoleted_by: [7230],
};

const readiness: ReadinessResponse = {
  ready: true,
  corpus_version: 1,
  chunks: 2744,
  dependencies: [],
};

const askFrames: AskEvent[] = [
  { type: "tool", tool: "search_rfcs", arguments: '{"query":"Host header"}', result_preview: "{}" },
  { type: "delta", text: "Yes. A client MUST send a Host header field (RFC 9110 Section 7.2)." },
  {
    type: "done",
    answer: "Yes. A client MUST send a Host header field (RFC 9110 Section 7.2).",
    citations: [
      {
        citation: "RFC 9110 Section 7.2",
        rfc_number: 9110,
        section_number: "7.2",
        source_url: "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2",
      },
    ],
    refused: false,
    hit_limit: false,
    trajectory: [
      { tool: "search_rfcs", arguments: '{"query":"Host header"}', result_preview: "{}" },
    ],
    usage: {
      prompt_tokens: 2000,
      completion_tokens: 120,
      total_tokens: 2120,
      tool_calls: 1,
      rounds: 2,
      cost_usd: 0.000992,
      elapsed_s: 4.2,
    },
  },
];

function sseResponse(frames: AskEvent[]) {
  const payload = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(payload));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
}

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (url.includes("/ready")) {
        return Promise.resolve(
          new Response(JSON.stringify(readiness), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url.includes("/api/ask")) return Promise.resolve(sseResponse(askFrames));
      return Promise.resolve(
        new Response(
          JSON.stringify({ query: "q", mode: "hybrid", total: 1, took_ms: 42, results: [hostChunk] }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/**
 * Run axe over the rendered document and return violations.
 *
 * WCAG 2.0/2.1 at A and AA, which is exactly the conformance the project claims - no
 * more, so a best-practice suggestion cannot fail a build over a standard nobody
 * promised, and no less.
 */
async function audit(): Promise<axe.AxeResults> {
  return axe.run(document.body, {
    runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
    // jsdom performs no layout, so these cannot be evaluated here. Contrast is covered
    // by contrast.test.ts against the shipped tokens.
    rules: { "color-contrast": { enabled: false } },
  });
}

async function violations(): Promise<axe.Result[]> {
  return (await audit()).violations;
}

function describeViolations(found: axe.Result[]): string {
  return found
    .map((v) => `${v.id} (${v.impact}): ${v.help}\n    ${v.nodes[0]?.html ?? ""}`)
    .join("\n  ");
}

async function renderApp() {
  render(<App />);
  await screen.findByText(/chunks indexed|corpus not ready|API unreachable/);
}

describe("axe audit", () => {
  it("actually evaluates rules, so a clean result means something", async () => {
    // Without this, a misconfigured runOnly or a selector that matches nothing would
    // report zero violations forever and the suite would look like it was passing.
    await renderApp();
    const results = await audit();
    expect(results.passes.length).toBeGreaterThan(10);
  });

  it("finds nothing on the landing state", async () => {
    await renderApp();
    const found = await violations();
    expect(found, describeViolations(found)).toHaveLength(0);
  });

  it("finds nothing while an answer is streaming", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.type(screen.getByRole("searchbox"), "is Host mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText(/A client MUST send a Host header field/);

    const found = await violations();
    expect(found, describeViolations(found)).toHaveLength(0);
  });

  it("finds nothing on a completed answer with sources", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.type(screen.getByRole("searchbox"), "is Host mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText("$0.0010")).toBeInTheDocument());

    const found = await violations();
    expect(found, describeViolations(found)).toHaveLength(0);
  });

  it("finds nothing on search results, including a superseded one", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.click(screen.getByRole("radio", { name: "Search" }));
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByRole("link", { name: /RFC 9110 Section 7.2/ });

    const found = await violations();
    expect(found, describeViolations(found)).toHaveLength(0);
  });

  it("finds nothing on the error state", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = typeof input === "string" ? input : (input as Request).url;
        return url.includes("/ready")
          ? Promise.resolve(new Response(JSON.stringify(readiness), { status: 200 }))
          : Promise.reject(new TypeError("network down"));
      }),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByRole("searchbox"), "host");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByRole("alert");

    const found = await violations();
    expect(found, describeViolations(found)).toHaveLength(0);
  });
});

describe("structure axe cannot check on its own", () => {
  it("has exactly one h1", async () => {
    await renderApp();
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });

  it("does not skip a heading level on an answered page", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.type(screen.getByRole("searchbox"), "is Host mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText("$0.0010")).toBeInTheDocument());

    const levels = screen
      .getAllByRole("heading")
      .map((h) => Number(h.tagName.slice(1)))
      .sort((a, b) => a - b);
    for (let i = 1; i < levels.length; i += 1) {
      expect(levels[i]! - levels[i - 1]!).toBeLessThanOrEqual(1);
    }
  });

  it("keeps the answer out of any live region", async () => {
    // Announcing a streamed answer token by token is unusable with a screen reader.
    const user = userEvent.setup();
    await renderApp();
    await user.type(screen.getByRole("searchbox"), "is Host mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    const answer = await screen.findByText(/A client MUST send a Host header field/);
    expect(answer.closest("[aria-live]")).toBeNull();
  });

  it("has exactly one status region, so announcements cannot race", async () => {
    await renderApp();
    expect(screen.getAllByRole("status")).toHaveLength(1);
  });

  it("gives every link an accessible name", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.click(screen.getByRole("radio", { name: "Search" }));
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByRole("link", { name: /RFC 9110 Section 7.2/ });

    for (const link of screen.getAllByRole("link")) {
      expect(link.textContent?.trim()).toBeTruthy();
    }
  });
});
