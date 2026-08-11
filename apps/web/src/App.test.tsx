import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { AskEvent, Citation, ReadinessResponse, SearchResponse } from "./lib/api";

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
  is_current: true,
  obsoleted_by: [],
};

const obsoleteChunk: Citation = {
  ...hostChunk,
  chunk_id: 2,
  rfc_number: 2616,
  title: "Hypertext Transfer Protocol -- HTTP/1.1",
  section_number: "14.23",
  section_title: "Host",
  citation: "RFC 2616 Section 14.23",
  source_url: "https://www.rfc-editor.org/rfc/rfc2616.html#section-14.23",
  is_current: false,
  obsoleted_by: [7230],
};

const readiness: ReadinessResponse = {
  ready: true,
  corpus_version: 1,
  chunks: 2744,
  dependencies: [],
};

/** `RequestInfo` may be a `Request`, which stringifies to "[object Object]". */
function urlOf(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  return input instanceof URL ? input.href : input.url;
}

const doneEvent: Extract<AskEvent, { type: "done" }> = {
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
};

const askFrames: AskEvent[] = [
  { type: "tool", tool: "search_rfcs", arguments: '{"query":"Host header"}', result_preview: "{}" },
  { type: "delta", text: "Yes. A client MUST send a " },
  { type: "delta", text: "Host header field (RFC 9110 Section 7.2)." },
  doneEvent,
];

/** An SSE body, optionally split at arbitrary byte offsets rather than frame edges. */
function sseResponse(frames: AskEvent[], splitInto = 1) {
  const payload = frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("");
  const size = Math.ceil(payload.length / splitInto);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (let at = 0; at < payload.length; at += size) {
        controller.enqueue(encoder.encode(payload.slice(at, at + size)));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function mockFetch(
  searchResponse: Partial<SearchResponse> = {},
  askResponse: () => Response = () => sseResponse(askFrames),
) {
  return vi.fn((input: RequestInfo | URL) => {
    const url = urlOf(input);
    if (url.includes("/ready")) {
      return Promise.resolve(
        new Response(JSON.stringify(readiness), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    }
    if (url.includes("/api/ask")) {
      return Promise.resolve(askResponse());
    }
    const body: SearchResponse = {
      query: "q",
      mode: "hybrid",
      total: 1,
      took_ms: 42.3,
      results: [hostChunk],
      ...searchResponse,
    };
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
  });
}

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch());
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/**
 * Render and wait for the readiness probe to settle.
 *
 * The probe resolves after mount, so rendering without awaiting it leaves a state
 * update outside act() that surfaces as a warning in whichever test runs next.
 */
async function renderApp() {
  render(<App />);
  await screen.findByText(/chunks indexed|corpus not ready|API unreachable/);
}

/** The app opens in ask mode, so retrieval tests have to ask for retrieval. */
async function switchToSearch(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("radio", { name: "Search" }));
}

describe("landing state", () => {
  it("shows the corpus size once readiness resolves", async () => {
    await renderApp();
    expect(await screen.findByText(/2,744 chunks indexed/)).toBeInTheDocument();
  });

  it("offers example queries before any search", async () => {
    await renderApp();
    expect(
      screen.getByRole("button", { name: "Is the Host header still mandatory?" }),
    ).toBeInTheDocument();
  });

  it("runs a search when an example is chosen", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.click(screen.getByRole("button", { name: /Which RFC obsoleted RFC 2616/ }));
    expect(await screen.findByRole("link", { name: /RFC 9110 Section 7.2/ })).toBeInTheDocument();
  });
});

describe("searching", () => {
  it("renders a citation linking to the exact section", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));

    const link = await screen.findByRole("link", { name: /RFC 9110 Section 7.2/ });
    expect(link).toHaveAttribute(
      "href",
      "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2",
    );
  });

  it("announces the result count in a live region", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));

    const status = await screen.findByRole("status");
    await waitFor(() => expect(status).toHaveTextContent(/1 result in 42 ms/));
  });

  it("keeps the search button disabled while the query is empty", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
  });

  it("shows an empty state rather than a blank page", async () => {
    vi.stubGlobal("fetch", mockFetch({ results: [], total: 0 }));
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.type(screen.getByRole("searchbox"), "zzzz");
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByText(/Nothing matched/)).toBeInTheDocument();
  });

  it("reports an unreachable API as an alert", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) =>
        urlOf(input).includes("/ready")
          ? Promise.resolve(new Response(JSON.stringify(readiness), { status: 200 }))
          : Promise.reject(new TypeError("network down")),
      ),
    );
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.type(screen.getByRole("searchbox"), "host");
    await user.click(screen.getByRole("button", { name: "Search" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not reach the API/);
  });
});

describe("supersession signalling", () => {
  it("marks obsoleted results and names the successor", async () => {
    vi.stubGlobal("fetch", mockFetch({ results: [obsoleteChunk] }));
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));

    const item = await screen.findByRole("listitem");
    expect(within(item).getByText("obsoleted")).toBeInTheDocument();
    expect(within(item).getByRole("link", { name: "RFC 7230" })).toBeInTheDocument();
  });
});

describe("accessibility", () => {
  it("exposes a skip link as the first tab stop", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.tab();
    expect(screen.getByRole("link", { name: /Skip to results/ })).toHaveFocus();
  });

  it("focuses the search box when / is pressed", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.keyboard("/");
    expect(screen.getByRole("searchbox")).toHaveFocus();
  });

  it("does not hijack / while the user is typing", async () => {
    const user = userEvent.setup();
    await renderApp();
    const box = screen.getByRole("searchbox");
    await user.click(box);
    await user.keyboard("application/json");
    expect(box).toHaveValue("application/json");
  });

  it("labels the retrieval mode group", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    expect(screen.getByRole("group", { name: /Retrieval/ })).toBeInTheDocument();
  });

  it("labels the mode group", async () => {
    await renderApp();
    expect(screen.getByRole("group", { name: /Mode/ })).toBeInTheDocument();
  });

  it("gives every filter an accessible name", async () => {
    const user = userEvent.setup();
    await renderApp();
    await switchToSearch(user);
    expect(screen.getByRole("checkbox", { name: /Current specs only/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Requirements only/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Hybrid" })).toBeChecked();
  });

  it("labels the search input for screen readers in each mode", async () => {
    const user = userEvent.setup();
    await renderApp();
    expect(screen.getByRole("searchbox", { name: /Ask a question/ })).toBeInTheDocument();
    await switchToSearch(user);
    expect(screen.getByRole("searchbox", { name: /Search the RFC corpus/ })).toBeInTheDocument();
  });

  it("does not announce the answer token by token", async () => {
    const user = userEvent.setup();
    await renderApp();
    await user.type(screen.getByRole("searchbox"), "is Host mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    // The answer itself must stay out of any live region; the status line carries the
    // one announcement. Announcing each delta would read the answer a word at a time.
    const answer = await screen.findByText(/A client MUST send a Host header field/);
    expect(answer.closest("[aria-live]")).toBeNull();
  });
});

describe("asking", () => {
  async function runAsk(user: ReturnType<typeof userEvent.setup>) {
    await user.type(screen.getByRole("searchbox"), "is the Host header mandatory");
    await user.click(screen.getByRole("button", { name: "Ask" }));
  }

  it("is the mode the app opens in", async () => {
    await renderApp();
    expect(screen.getByRole("radio", { name: "Ask" })).toBeChecked();
  });

  it("streams an answer and lists its sources", async () => {
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);

    expect(
      await screen.findByText(/Yes\. A client MUST send a Host header field/),
    ).toBeInTheDocument();
    const source = await screen.findByRole("link", { name: /RFC 9110 Section 7\.2/ });
    expect(source).toHaveAttribute(
      "href",
      "https://www.rfc-editor.org/rfc/rfc9110.html#section-7.2",
    );
  });

  it("shows which tools produced the answer", async () => {
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);
    expect(await screen.findByText(/Searched the corpus: “Host header”/)).toBeInTheDocument();
  });

  it("reassembles frames split across reads", async () => {
    // The stream arrives in fixed-size pieces that cut through the middle of frames,
    // which is what a real network does and what the frame buffer exists to survive.
    vi.stubGlobal("fetch", mockFetch({}, () => sseResponse(askFrames, 17)));
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);
    expect(
      await screen.findByText(/Yes\. A client MUST send a Host header field/),
    ).toBeInTheDocument();
  });

  it("discards text the server withdraws", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        sseResponse([
          { type: "delta", text: "Let me look that up." },
          { type: "reset", reason: "tool_preamble" },
          { type: "delta", text: "Yes. A client MUST send a Host header field (RFC 9110)." },
          { ...doneEvent, answer: "Yes. A client MUST send a Host header field (RFC 9110)." },
        ]),
      ),
    );
    const user = userEvent.setup();
    render(<App />);
    await runAsk(user);

    expect(await screen.findByText(/Yes\. A client MUST send/)).toBeInTheDocument();
    expect(screen.queryByText(/Let me look that up/)).not.toBeInTheDocument();
  });

  it("reports what the answer cost", async () => {
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);
    expect(await screen.findByText("$0.0010")).toBeInTheDocument();
    expect(screen.getByText("4.2s")).toBeInTheDocument();
  });

  it("announces completion once, in the status region", async () => {
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);
    const status = await screen.findByRole("status");
    await waitFor(() =>
      expect(status).toHaveTextContent(/Answer complete, 1 source cited, in 4.2 seconds/),
    );
  });

  it("marks an answer the corpus could not support", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        sseResponse([
          { type: "delta", text: "Status code 599 is not defined in RFC 9110." },
          {
            ...doneEvent,
            answer: "Status code 599 is not defined in RFC 9110.",
            refused: true,
          },
        ]),
      ),
    );
    const user = userEvent.setup();
    render(<App />);
    await runAsk(user);
    expect(await screen.findByText("no answer in the corpus")).toBeInTheDocument();
  });

  it("surfaces an in-band error where the answer would be", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(sseResponse([{ type: "error", message: "no active corpus version" }])),
    );
    const user = userEvent.setup();
    render(<App />);
    await runAsk(user);
    expect(await screen.findByRole("alert")).toHaveTextContent(/no active corpus version/);
  });

  it("clears the answer when the user switches to search", async () => {
    const user = userEvent.setup();
    await renderApp();
    await runAsk(user);
    await screen.findByText(/Yes\. A client MUST send/);

    await switchToSearch(user);
    expect(screen.queryByText(/Yes\. A client MUST send/)).not.toBeInTheDocument();
  });
});
