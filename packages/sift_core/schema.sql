-- Sift schema.
--
-- ${VECTOR_TYPE} is substituted at migration time (see db.py) because the embedding
-- dimension is one of the variables the optimization sweep changes, and pgvector
-- needs a fixed dimension before it can build an index.

CREATE EXTENSION IF NOT EXISTS vector;

-- Every RFC in the index, not only the embedded ones.
--
-- Storing all ~9,800 costs almost nothing without vectors, and it is what lets
-- resolve_current_spec() walk a supersession chain through documents that were never
-- selected for embedding: RFC 2616 -> 7230 -> 9110 resolves even though 7230 itself
-- is not in the corpus.
CREATE TABLE IF NOT EXISTS documents (
    number       integer PRIMARY KEY,
    title        text        NOT NULL,
    year         integer     NOT NULL DEFAULT 0,
    status       text        NOT NULL,
    stream       text,
    page_count   integer     NOT NULL DEFAULT 0,
    abstract     text,
    authors      text[]      NOT NULL DEFAULT '{}',
    keywords     text[]      NOT NULL DEFAULT '{}',
    area         text,
    wg           text,
    obsoletes    integer[]   NOT NULL DEFAULT '{}',
    obsoleted_by integer[]   NOT NULL DEFAULT '{}',
    updates      integer[]   NOT NULL DEFAULT '{}',
    updated_by   integer[]   NOT NULL DEFAULT '{}',
    has_text     boolean     NOT NULL DEFAULT true,
    is_embedded  boolean     NOT NULL DEFAULT false,
    ingested_at  timestamptz
);

CREATE INDEX IF NOT EXISTS documents_status_idx   ON documents (status);
CREATE INDEX IF NOT EXISTS documents_year_idx     ON documents (year);
CREATE INDEX IF NOT EXISTS documents_embedded_idx ON documents (is_embedded) WHERE is_embedded;
-- GIN over the supersession edges keeps "what obsoleted X" a single indexed lookup.
CREATE INDEX IF NOT EXISTS documents_obsoleted_by_idx ON documents USING gin (obsoleted_by);
CREATE INDEX IF NOT EXISTS documents_updates_idx      ON documents USING gin (updates);

-- One row per ingest configuration, so every number in the results table can be
-- traced back to the exact chunking and embedding settings that produced it.
CREATE TABLE IF NOT EXISTS corpus_versions (
    id              serial PRIMARY KEY,
    fingerprint     text        NOT NULL,
    embedding_model text        NOT NULL,
    dimensions      integer     NOT NULL,
    chunk_config    jsonb       NOT NULL,
    is_active       boolean     NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    notes           text,
    UNIQUE (fingerprint, embedding_model, dimensions)
);

-- Only one configuration serves queries at a time.
CREATE UNIQUE INDEX IF NOT EXISTS corpus_versions_one_active
    ON corpus_versions ((is_active)) WHERE is_active;

CREATE TABLE IF NOT EXISTS chunks (
    id             bigserial PRIMARY KEY,
    version_id     integer NOT NULL REFERENCES corpus_versions (id) ON DELETE CASCADE,
    rfc_number     integer NOT NULL REFERENCES documents (number)   ON DELETE CASCADE,
    ordinal        integer NOT NULL,
    section_number text,
    section_title  text,
    text           text    NOT NULL,
    embedding      ${VECTOR_TYPE},
    has_normative  boolean NOT NULL DEFAULT false,
    char_start     integer NOT NULL DEFAULT 0,
    char_end       integer NOT NULL DEFAULT 0,
    content_hash   bytea   NOT NULL,
    metadata       jsonb   NOT NULL DEFAULT '{}',
    -- Section titles carry heavily weighted terms ("Host and :authority"), so the
    -- keyword half of hybrid search indexes them alongside the body. Weight A marks
    -- the title so ts_rank_cd favours a title match over an incidental body mention.
    tsv tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(section_title, '')), 'A') ||
        setweight(to_tsvector('english', text), 'B')
    ) STORED,
    UNIQUE (version_id, rfc_number, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx     ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_rfc_idx     ON chunks (version_id, rfc_number);
CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks (version_id, rfc_number, section_number);
CREATE INDEX IF NOT EXISTS chunks_normative_idx
    ON chunks (version_id) WHERE has_normative;

-- The HNSW index is deliberately NOT created here. Building it before a bulk load
-- makes ingestion far slower and yields a worse graph than building it once the rows
-- are in place, so create_vector_index() is called after ingestion completes.
