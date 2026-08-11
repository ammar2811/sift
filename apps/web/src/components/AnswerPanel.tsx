import type { AgentUsage, AnswerCitation, ToolStep } from "../lib/api";

/**
 * What each tool means in the reader's terms. The agent's own names are accurate but
 * they describe the implementation, and the point of showing the trajectory is to let
 * someone judge whether the answer was retrieved or invented.
 */
const TOOL_LABELS: Record<string, string> = {
  search_rfcs: "Searched the corpus",
  get_rfc_metadata: "Checked RFC metadata",
  get_section: "Read a section verbatim",
  resolve_current_spec: "Followed the supersession chain",
};

function describeStep(step: ToolStep): string {
  const label = TOOL_LABELS[step.tool] ?? step.tool;
  try {
    const args = JSON.parse(step.arguments) as Record<string, unknown>;
    if (typeof args["query"] === "string") return `${label}: “${args["query"]}”`;
    if (typeof args["rfc_number"] === "number") {
      const section = typeof args["section"] === "string" ? ` §${args["section"]}` : "";
      return `${label}: RFC ${args["rfc_number"]}${section}`;
    }
  } catch {
    /* arguments are model output; a malformed object is not worth failing over */
  }
  return label;
}

interface Props {
  answer: string;
  citations: AnswerCitation[];
  trajectory: ToolStep[];
  usage: AgentUsage | null;
  refused: boolean;
  streaming: boolean;
}

export function AnswerPanel({
  answer,
  citations,
  trajectory,
  usage,
  refused,
  streaming,
}: Props) {
  return (
    <div className="answer-panel">
      {trajectory.length > 0 && (
        <ol className="trajectory">
          {trajectory.map((step, index) => (
            <li key={`${step.tool}-${index}`}>{describeStep(step)}</li>
          ))}
        </ol>
      )}

      {/* Not a live region. Announcing every token would make the page unusable with a
          screen reader; the status line above announces the answer once it settles. */}
      <article className="answer" aria-busy={streaming}>
        {refused && !streaming && (
          <p className="badge-row">
            <span className="badge caution">no answer in the corpus</span>
          </p>
        )}
        <p className="answer-body">
          {answer}
          {streaming && <span className="caret" aria-hidden="true" />}
        </p>
      </article>

      {citations.length > 0 && (
        <section className="citations" aria-label="Sources">
          <h2 className="citations-heading">Sources</h2>
          <ul>
            {citations.map((citation) => (
              <li key={citation.citation}>
                <a href={citation.source_url} target="_blank" rel="noreferrer">
                  {citation.citation}
                  <span className="visually-hidden">
                    {" "}
                    (opens on rfc-editor.org in a new tab)
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}

      {usage && (
        <dl className="usage">
          <dt>time</dt>
          <dd>{usage.elapsed_s.toFixed(1)}s</dd>
          <dt>tool calls</dt>
          <dd>{usage.tool_calls}</dd>
          <dt>tokens</dt>
          <dd>{usage.total_tokens.toLocaleString()}</dd>
          <dt>cost</dt>
          <dd>${usage.cost_usd.toFixed(4)}</dd>
        </dl>
      )}
    </div>
  );
}
