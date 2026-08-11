/**
 * Typed client for the Sift API.
 *
 * Types are declared here rather than generated so the shape the UI depends on is
 * explicit and reviewable; they mirror `apps/api/schemas.py`.
 */

export type SearchMode = "dense" | "keyword" | "hybrid";

export interface Citation {
  chunk_id: number;
  rfc_number: number;
  title: string;
  section_number: string | null;
  section_title: string | null;
  text: string;
  citation: string;
  source_url: string;
  score: number;
  dense_rank: number | null;
  keyword_rank: number | null;
  is_current: boolean;
  obsoleted_by: number[];
}

export interface SearchResponse {
  query: string;
  mode: string;
  total: number;
  took_ms: number;
  results: Citation[];
}

export interface SearchRequest {
  query: string;
  k?: number;
  mode?: SearchMode;
  current_only?: boolean;
  normative_only?: boolean;
  rfc_numbers?: number[];
  min_year?: number | null;
}

export interface DependencyStatus {
  name: string;
  ok: boolean;
  detail: string | null;
  latency_ms: number | null;
}

export interface ReadinessResponse {
  ready: boolean;
  corpus_version: number | null;
  chunks: number | null;
  dependencies: DependencyStatus[];
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    // FastAPI reports errors as {detail: ...}; fall back to the status text when the
    // body is not JSON, which happens for proxy and gateway failures.
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function search(body: SearchRequest, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}

export function getReadiness(signal?: AbortSignal): Promise<ReadinessResponse> {
  return request<ReadinessResponse>("/ready", { signal });
}

export interface AnswerCitation {
  citation: string;
  rfc_number: number;
  section_number: string | null;
  source_url: string;
}

export interface ToolStep {
  tool: string;
  arguments: string;
  result_preview: string;
}

export interface AgentUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tool_calls: number;
  rounds: number;
  cost_usd: number;
  elapsed_s: number;
}

/**
 * One frame of an answer.
 *
 * `reset` is the one that shapes the consumer: tokens are streamed before the server
 * knows whether they are the final answer, so anything already drawn must be dropped
 * when it arrives. A client that ignores it renders a discarded draft followed by the
 * real answer.
 */
export type AskEvent =
  | { type: "tool"; tool: string; arguments: string; result_preview: string }
  | { type: "delta"; text: string }
  | { type: "reset"; reason: string }
  | {
      type: "done";
      answer: string;
      citations: AnswerCitation[];
      refused: boolean;
      hit_limit: boolean;
      trajectory: ToolStep[];
      usage: AgentUsage;
    }
  | { type: "error"; message: string };

/**
 * Stream an answer, invoking `onEvent` per frame.
 *
 * Uses fetch over EventSource because the question goes in a POST body and
 * EventSource can only issue GETs.
 */
export async function ask(
  body: { query: string },
  onEvent: (event: AskEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const parsed = (await response.json()) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  if (!response.body) throw new ApiError("this browser cannot read a streamed response", 0);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    // A frame can be split across reads, so decode with `stream` and only consume
    // whole frames - the trailing partial one stays buffered for the next read.
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of frame.split("\n")) {
        if (line.startsWith("data: ")) {
          onEvent(JSON.parse(line.slice(6)) as AskEvent);
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
