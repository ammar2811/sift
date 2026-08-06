import { useCallback, useEffect, useId, useRef, useState } from "react";

import { ResultCard } from "./components/ResultCard";
import {
  ApiError,
  getReadiness,
  search,
  type Citation,
  type ReadinessResponse,
  type SearchMode,
} from "./lib/api";

const MODES: { value: SearchMode; label: string; hint: string }[] = [
  { value: "hybrid", label: "Hybrid", hint: "Vector and keyword results fused by rank" },
  { value: "dense", label: "Vector", hint: "Semantic similarity only" },
  { value: "keyword", label: "Keyword", hint: "Full-text search only" },
];

const EXAMPLES = [
  "Is the Host header still mandatory?",
  "What must a client do when it receives a 417 response?",
  "Which RFC obsoleted RFC 2616?",
  "How are DNS label lengths limited?",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [currentOnly, setCurrentOnly] = useState(false);
  const [normativeOnly, setNormativeOnly] = useState(false);
  const [showInspector, setShowInspector] = useState(true);

  const [results, setResults] = useState<Citation[] | null>(null);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const inFlight = useRef<AbortController | null>(null);
  const modeGroupId = useId();

  useEffect(() => {
    const controller = new AbortController();
    getReadiness(controller.signal)
      .then(setReadiness)
      .catch(() => setReadiness(null));
    return () => controller.abort();
  }, []);

  // "/" focuses search, the convention on documentation sites - but never while the
  // user is already typing somewhere.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target?.isContentEditable;
      if (event.key === "/" && !typing) {
        event.preventDefault();
        inputRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const runSearch = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;

      // A superseded request must not overwrite a newer one's results.
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;

      setLoading(true);
      setError(null);
      try {
        const response = await search(
          {
            query: trimmed,
            mode,
            k: 10,
            current_only: currentOnly,
            normative_only: normativeOnly,
          },
          controller.signal,
        );
        setResults(response.results);
        setTookMs(response.took_ms);
      } catch (err) {
        if (controller.signal.aborted) return;
        setResults(null);
        setError(
          err instanceof ApiError
            ? `Search failed: ${err.message}`
            : "Could not reach the API. Is the server running?",
        );
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    },
    [mode, currentOnly, normativeOnly],
  );

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    void runSearch(query);
  }

  function onExample(text: string) {
    setQuery(text);
    void runSearch(text);
    inputRef.current?.focus();
  }

  const ready = readiness?.ready ?? false;
  const statusClass = readiness === null ? "down" : ready ? "ready" : "";

  return (
    <>
      <a className="skip-link" href="#results">
        Skip to results
      </a>

      <div className="app">
        <header className="masthead">
          <div>
            <h1 className="wordmark">
              Sift<span>.</span>
            </h1>
            <p className="tagline">
              Search IETF RFCs with section-precise citations.
            </p>
          </div>
          <p className="corpus-badge">
            <span className={`dot ${statusClass}`} aria-hidden="true" />
            {readiness === null
              ? "API unreachable"
              : ready
                ? `${readiness.chunks?.toLocaleString()} chunks indexed`
                : "corpus not ready"}
          </p>
        </header>

        <main>
          <form className="search" onSubmit={onSubmit} role="search">
            <div className="search-row">
              <label className="visually-hidden" htmlFor="q">
                Search the RFC corpus
              </label>
              <input
                id="q"
                ref={inputRef}
                className="search-input"
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about a protocol requirement…  (press / to focus)"
                autoComplete="off"
                spellCheck={false}
              />
              <button className="btn" type="submit" disabled={loading || !query.trim()}>
                {loading ? "Searching…" : "Search"}
              </button>
            </div>

            <div className="controls">
              <div className="control-group" role="group" aria-labelledby={modeGroupId}>
                <span className="group-label" id={modeGroupId}>
                  Retrieval
                </span>
                <div className="segmented">
                  {MODES.map((option) => (
                    <label key={option.value} title={option.hint}>
                      <input
                        type="radio"
                        name="mode"
                        value={option.value}
                        checked={mode === option.value}
                        onChange={() => setMode(option.value)}
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={currentOnly}
                  onChange={(e) => setCurrentOnly(e.target.checked)}
                />
                Current specs only
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={normativeOnly}
                  onChange={(e) => setNormativeOnly(e.target.checked)}
                />
                Requirements only
              </label>

              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={showInspector}
                  onChange={(e) => setShowInspector(e.target.checked)}
                />
                Show retrieval detail
              </label>
            </div>
          </form>

          {/* Announced to screen readers whenever the result count changes, so the
              outcome of a search is not silent for non-visual users. */}
          <p className="status-line" role="status" aria-live="polite">
            {loading
              ? "Searching…"
              : results
                ? `${results.length} result${results.length === 1 ? "" : "s"} in ${tookMs?.toFixed(0)} ms`
                : ""}
          </p>

          <div id="results" ref={resultsRef} tabIndex={-1}>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}

            {loading && !results && (
              <div className="results" aria-hidden="true">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="skeleton" />
                ))}
              </div>
            )}

            {results && results.length > 0 && (
              <ul className="results">
                {results.map((hit) => (
                  <ResultCard key={hit.chunk_id} hit={hit} showInspector={showInspector} />
                ))}
              </ul>
            )}

            {results && results.length === 0 && !loading && (
              <div className="empty">
                <p style={{ margin: 0 }}>
                  Nothing matched. Try fewer filters, or the keyword mode for an exact
                  phrase.
                </p>
              </div>
            )}

            {!results && !error && !loading && (
              <div className="empty">
                <p style={{ marginTop: 0 }}>
                  Search across IETF standards. Every result cites an exact section.
                </p>
                <ul className="examples">
                  {EXAMPLES.map((text) => (
                    <li key={text}>
                      <button
                        type="button"
                        className="example-btn"
                        onClick={() => onExample(text)}
                      >
                        {text}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </main>
      </div>
    </>
  );
}
