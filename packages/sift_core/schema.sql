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
    -- Which document set was ingested ('sweep', 'full', ...). Part of the key
    -- because a metric is only comparable against the same documents: without it,
    -- ingesting the full corpus under the same chunking config would silently merge
    -- into the sweep corpus's version and invalidate every result recorded against it.
    scope           text        NOT NULL DEFAULT 'default',
    chunk_config    jsonb       NOT NULL,
    is_active       boolean     NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now(),
    notes           text
);

-- Added defensively so an existing database picks up the column rather than needing a
-- rebuild; CREATE TABLE IF NOT EXISTS silently skips a table that already exists.
ALTER TABLE corpus_versions ADD COLUMN IF NOT EXISTS scope text NOT NULL DEFAULT 'default';
ALTER TABLE corpus_versions
    DROP CONSTRAINT IF EXISTS corpus_versions_fingerprint_embedding_model_dimensions_key;
CREATE UNIQUE INDEX IF NOT EXISTS corpus_versions_identity
    ON corpus_versions (fingerprint, embedding_model, dimensions, scope);

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

-- How many chunks contain each word, per corpus version.
--
-- PostgreSQL's ts_rank_cd has no inverse-document-frequency term: a match on "417",
-- which occurs in 12 of 129,109 chunks, counts exactly as much as a match on
-- "response". At 7,364 chunks that rarely changed an outcome; at 129,109 it decides
-- them, and the correct chunk for "what must a client do on a 417 response" scored
-- 1.40 against 9.20 for a DHCP section that merely repeats common words.
--
-- Storing document frequencies lets the query side drop words that cannot discriminate
-- before ranking ever happens, which is the part BM25 would otherwise supply.
CREATE TABLE IF NOT EXISTS lexeme_stats (
    version_id integer NOT NULL REFERENCES corpus_versions (id) ON DELETE CASCADE,
    lexeme     text    NOT NULL,
    ndoc       integer NOT NULL,
    PRIMARY KEY (version_id, lexeme)
);

CREATE INDEX IF NOT EXISTS lexeme_stats_ndoc_idx ON lexeme_stats (version_id, ndoc);

-- The HNSW index is deliberately NOT created here. Building it before a bulk load
-- makes ingestion far slower and yields a worse graph than building it once the rows
-- are in place, so create_vector_index() is called after ingestion completes.
