import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { Citation, ReadinessResponse, SearchResponse } from "./lib/api";

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

function mockFetch(searchResponse: Partial<SearchResponse> = {}) {
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
    await user.click(screen.getByRole("button", { name: /Which RFC obsoleted RFC 2616/ }));
    expect(await screen.findByRole("link", { name: /RFC 9110 Section 7.2/ })).toBeInTheDocument();
  });
});

describe("searching", () => {
  it("renders a citation linking to the exact section", async () => {
    const user = userEvent.setup();
    await renderApp();
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
    await user.type(screen.getByRole("searchbox"), "host header");
    await user.click(screen.getByRole("button", { name: "Search" }));

    const status = await screen.findByRole("status");
    await waitFor(() => expect(status).toHaveTextContent(/1 result in 42 ms/));
  });

  it("keeps the search button disabled while the query is empty", async () => {
    await renderApp();
    expect(screen.getByRole("button", { name: "Search" })).toBeDisabled();
  });

  it("shows an empty state rather than a blank page", async () => {
    vi.stubGlobal("fetch", mockFetch({ results: [], total: 0 }));
    const user = userEvent.setup();
    await renderApp();
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
    await renderApp();
    expect(screen.getByRole("group", { name: /Retrieval/ })).toBeInTheDocument();
  });

  it("gives every filter an accessible name", async () => {
    await renderApp();
    expect(screen.getByRole("checkbox", { name: /Current specs only/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Requirements only/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Hybrid" })).toBeChecked();
  });

  it("labels the search input for screen readers", async () => {
    await renderApp();
    expect(screen.getByRole("searchbox", { name: /Search the RFC corpus/ })).toBeInTheDocument();
  });
});
