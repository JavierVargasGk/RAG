
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS doc_chunks (
    id SERIAL PRIMARY KEY, 
    content TEXT NOT NULL,
    filename TEXT,
    embedding VECTOR(1024),
    page_number INTEGER
);

CALL paradedb.create_bm25(
    index_name => 'search_idx',
    table_name => 'doc_chunks',
    column_name => 'content',
    key_field => 'id'
);