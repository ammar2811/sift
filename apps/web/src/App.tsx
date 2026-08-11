import { useCallback, useEffect, useId, useRef, useState } from "react";

import { AnswerPanel } from "./components/AnswerPanel";
import { ResultCard } from "./components/ResultCard";
import {
  ApiError,
  ask,
  getReadiness,
  search,
  type AgentUsage,
  type AnswerCitation,
  type Citation,
  type ReadinessResponse,
  type SearchMode,
  type ToolStep,
} from "./lib/api";

type Task = "ask" | "search";

const TASKS: { value: Task; label: string; hint: string }[] = [
  { value: "ask", label: "Ask", hint: "A cited answer, assembled from the corpus" },
  { value: "search", label: "Search", hint: "The retrieved passages themselves" },
];

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
  const [task, setTask] = useState<Task>("ask");
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

  const [answer, setAnswer] = useState<string | null>(null);
  const [citations, setCitations] = useState<AnswerCitation[]>([]);
  const [trajectory, setTrajectory] = useState<ToolStep[]>([]);
  const [usage, setUsage] = useState<AgentUsage | null>(null);
  const [refused, setRefused] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const inFlight = useRef<AbortController | null>(null);
  const modeGroupId = useId();
  const taskGroupId = useId();

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

  const runAsk = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setLoading(true);
    setError(null);
    setAnswer("");
    setCitations([]);
    setTrajectory([]);
    setUsage(null);
    setRefused(false);

    try {
      await ask(
        { query: trimmed },
        (event) => {
          switch (event.type) {
            case "delta":
              setAnswer((previous) => (previous ?? "") + event.text);
              break;
            // Text drawn before the loop knew it was not the answer.
            case "reset":
              setAnswer("");
              break;
            case "tool":
              setTrajectory((previous) => [...previous, event]);
              break;
            case "done":
              // Prefer the assembled answer over the accumulated deltas: they should
              // agree, and if they ever do not, the server's copy is the real one.
              setAnswer(event.answer);
              setCitations(event.citations);
              setTrajectory(event.trajectory);
              setUsage(event.usage);
              setRefused(event.refused);
              break;
            case "error":
              setError(event.message);
              break;
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      setAnswer(null);
      setError(
        err instanceof ApiError
          ? `Could not answer: ${err.message}`
          : "Could not reach the API. Is the server running?",
      );
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  const run = useCallback(
    (text: string) => (task === "ask" ? runAsk(text) : runSearch(text)),
    [task, runAsk, runSearch],
  );

  function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    void run(query);
  }

  function onExample(text: string) {
    setQuery(text);
    void run(text);
    inputRef.current?.focus();
  }

  // Switching task discards the other task's output rather than leaving a stale answer
  // sitting above a fresh set of search results.
  function onTaskChange(next: Task) {
    inFlight.current?.abort();
    setTask(next);
    setLoading(false);
    setError(null);
    setResults(null);
    setAnswer(null);
    setCitations([]);
    setTrajectory([]);
    setUsage(null);
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
              Ask IETF RFCs a question. Every answer cites an exact section.
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
                {task === "ask" ? "Ask a question about the RFCs" : "Search the RFC corpus"}
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
                {loading
                  ? task === "ask"
                    ? "Answering…"
                    : "Searching…"
                  : task === "ask"
                    ? "Ask"
                    : "Search"}
              </button>
            </div>

            <div className="controls">
              <div className="control-group" role="group" aria-labelledby={taskGroupId}>
                <span className="group-label" id={taskGroupId}>
                  Mode
                </span>
                <div className="segmented">
                  {TASKS.map((option) => (
                    <label key={option.value} title={option.hint}>
                      <input
                        type="radio"
                        name="task"
                        value={option.value}
                        checked={task === option.value}
                        onChange={() => onTaskChange(option.value)}
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Retrieval controls tune what search returns. In ask mode the agent
                  chooses its own retrieval per tool call, so showing them would imply
                  a control the user does not have. */}
              {task === "search" && (
                <>
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
                </>
              )}
            </div>
          </form>

          {/* Announced to screen readers whenever the result count changes, so the
              outcome of a search is not silent for non-visual users. */}
          {/* The single announcement point. In ask mode this is what tells a screen
              reader the answer has settled, since streaming the tokens themselves into
              a live region would announce the answer a word at a time. */}
          <p className="status-line" role="status" aria-live="polite">
            {loading
              ? task === "ask"
                ? "Working on an answer…"
                : "Searching…"
              : task === "ask"
                ? usage
                  ? `Answer complete, ${citations.length} source${citations.length === 1 ? "" : "s"} cited, in ${usage.elapsed_s.toFixed(1)} seconds`
                  : ""
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

            {task === "ask" && answer !== null && (
              <AnswerPanel
                answer={answer}
                citations={citations}
                trajectory={trajectory}
                usage={usage}
                refused={refused}
                streaming={loading}
              />
            )}

            {task === "search" && loading && !results && (
              <div className="results" aria-hidden="true">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="skeleton" />
                ))}
              </div>
            )}

            {task === "search" && results && results.length > 0 && (
              <ul className="results">
                {results.map((hit) => (
                  <ResultCard key={hit.chunk_id} hit={hit} showInspector={showInspector} />
                ))}
              </ul>
            )}

            {task === "search" && results && results.length === 0 && !loading && (
              <div className="empty">
                <p style={{ margin: 0 }}>
                  Nothing matched. Try fewer filters, or the keyword mode for an exact
                  phrase.
                </p>
              </div>
            )}

            {!results && answer === null && !error && !loading && (
              <div className="empty">
                <p style={{ marginTop: 0 }}>
                  {task === "ask"
                    ? "Ask a question about any IETF standard. Answers are assembled only from the corpus, and cite the sections they came from."
                    : "Search across IETF standards. Every result cites an exact section."}
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
