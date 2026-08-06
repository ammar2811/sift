import type { Citation } from "../lib/api";

/**
 * RFC bodies mix prose with ASCII diagrams, ABNF grammars and packet layouts. Those
 * only make sense in a monospaced block that scrolls horizontally rather than
 * wrapping, so the shape of the text decides how it is rendered.
 */
function looksPreformatted(text: string): boolean {
  const lines = text.split("\n");
  if (lines.length < 3) return false;

  const structural = lines.filter((line) => {
    const trimmed = line.trim();
    if (!trimmed) return false;
    // Box drawing, table rules, bit-field diagrams, or ABNF assignment.
    return (
      /^[|+\\/\-_= ]{6,}$/.test(trimmed) ||
      /\|.*\|/.test(trimmed) ||
      /^\S+\s*=\s*\S/.test(trimmed)
    );
  }).length;

  return structural / lines.filter((l) => l.trim()).length > 0.4;
}

interface Props {
  hit: Citation;
  showInspector: boolean;
}

export function ResultCard({ hit, showInspector }: Props) {
  const preformatted = looksPreformatted(hit.text);
  const headingId = `result-${hit.chunk_id}-heading`;

  return (
    <li className="result">
      <article aria-labelledby={headingId}>
        <div className="result-head">
          <h3 id={headingId} style={{ margin: 0, fontSize: "1rem" }}>
            <a
              className="citation-link"
              href={hit.source_url}
              target="_blank"
              rel="noreferrer"
            >
              {hit.citation}
              <span className="visually-hidden"> (opens on rfc-editor.org in a new tab)</span>
            </a>
          </h3>
          {!hit.is_current && (
            <span className="badge obsolete">obsoleted</span>
          )}
        </div>

        <p className="result-title">
          {hit.title}
          {hit.section_title ? ` — ${hit.section_title}` : ""}
        </p>

        <p className={preformatted ? "result-body preformatted" : "result-body"}>
          {hit.text}
        </p>

        {!hit.is_current && hit.obsoleted_by.length > 0 && (
          <p className="supersession">
            Superseded by{" "}
            {hit.obsoleted_by.map((number, index) => (
              <span key={number}>
                {index > 0 && ", "}
                <a
                  href={`https://www.rfc-editor.org/rfc/rfc${number}.html`}
                  target="_blank"
                  rel="noreferrer"
                >
                  RFC {number}
                </a>
              </span>
            ))}
            . Check the current specification before relying on this text.
          </p>
        )}

        {showInspector && (
          <details className="inspector">
            <summary>retrieval detail</summary>
            <dl className="rank-grid">
              <dt>score</dt>
              <dd>{hit.score.toFixed(5)}</dd>
              <dt>vector rank</dt>
              <dd>{hit.dense_rank ?? "not retrieved"}</dd>
              <dt>keyword rank</dt>
              <dd>{hit.keyword_rank ?? "not retrieved"}</dd>
              <dt>chunk id</dt>
              <dd>{hit.chunk_id}</dd>
            </dl>
          </details>
        )}
      </article>
    </li>
  );
}
