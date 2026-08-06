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

export interface DocumentSummary {
  number: number;
  title: string;
  year: number;
  status: string;
  abstract: string | null;
  authors: string[];
  area: string | null;
  wg: string | null;
  is_current: boolean;
  is_embedded: boolean;
  obsoletes: number[];
  obsoleted_by: number[];
  updates: number[];
  updated_by: number[];
  source_url: string;
}

export interface SupersessionChain {
  requested: number;
  current: number;
  is_current: boolean;
  chain: DocumentSummary[];
  note: string | null;
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

export function getCurrentSpec(
  rfcNumber: number,
  signal?: AbortSignal,
): Promise<SupersessionChain> {
  return request<SupersessionChain>(`/api/documents/${rfcNumber}/current`, { signal });
}
