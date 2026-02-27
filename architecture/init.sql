
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
DROP TABLE doc_chunks;

CREATE TABLE doc_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    filename TEXT,
    embedding vector(1024),
    page_number INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bm25 ON doc_chunks 
USING bm25 (id, content, filename) 
WITH (key_field = 'id');